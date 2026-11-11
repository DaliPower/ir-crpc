"""Synthetic restart/count orchestration checks; no experiment is dispatched."""
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import zipfile

import pytest

from experiments import revision_continue_pipeline as pipeline
from experiments import revision_provenance
from manuscript_revision import revision_fair_q_selection as fair


def install_module(monkeypatch, name, **attributes):
    value = ModuleType(name)
    value.__dict__.update(attributes)
    monkeypatch.setitem(sys.modules, name, value)
    return value


@pytest.fixture
def restart(tmp_path, monkeypatch):
    real_root = pipeline.ROOT
    root = tmp_path / 'repository'
    base = root / 'exp/revision_access_41572'
    model = root / 'models/model.py'
    model.parent.mkdir(parents=True)
    model.write_text('# corrected synthetic development source\n', encoding='utf-8')
    (root / 'aux.py').write_text('# synthetic immutable auxiliary\n', encoding='utf-8')
    source = {'models/model.py': pipeline.digest(model)}
    archive_dir = base / 'protocol/source_snapshots'
    archive_dir.mkdir(parents=True)
    archive = archive_dir / 'development_02.zip'
    with zipfile.ZipFile(archive, 'x') as handle:
        handle.writestr('models/model.py', model.read_bytes())
        handle.writestr('training_source_manifest.json', json.dumps({'source_sha256': source}))
    archive_path = archive.with_suffix('.json')
    pipeline.write_new(archive_path, {'source': {'source_sha256': source}, 'archive': str(archive),
                                      'sha256': pipeline.digest(archive)})
    verification = base / 'source_correction/verification.xml'
    verification.parent.mkdir(parents=True)
    verification.write_text('<testsuite tests="400" failures="0" errors="0" skipped="0"/>')
    resume_evidence = verification.with_name('real_resume.json')
    pipeline.write_new(resume_evidence, {'status': 'passed', 'synthetic_fixture_only': True})
    revision = base / 'source_correction/source_revision.json'
    pipeline.write_new(revision, {
        'status': 'verified_full_development_restart',
        'training_source_before': {'models/model.py': '0' * 64}, 'training_source_after': source,
        'changed_training_files': ['models/model.py'], 'old_results_used_for_selection': False,
        'same_declared_plans_seeds_and_budgets': True, 'planned_development_runs': 150,
        'evidence_files': [{'path': str(path), 'sha256': pipeline.digest(path)}
                           for path in (verification, resume_evidence)],
        'verification_xml': str(verification), 'real_resume_evidence': str(resume_evidence),
    })
    results = base / 'development_v02'
    registry_path = base / 'protocol/dev_fair_q_v02/fair_q_registry.json'
    fair.bind_results(real_root / 'exp/revision_access_41572/protocol/dev_fair_q_01/fair_q_registry.json',
                      results, archive_path, registry_path)
    monkeypatch.setattr(pipeline, 'ROOT', root)
    monkeypatch.setattr(pipeline, 'BASE', base)
    monkeypatch.setattr(pipeline, 'AUXILIARY_FILES', ('aux.py',))
    monkeypatch.setattr(revision_provenance, 'training_source_manifest',
                        lambda: {'source_sha256': {'models/model.py': pipeline.digest(model)}})
    return SimpleNamespace(root=root, base=base, source=source, model=model, archive=archive_path,
                           verification=verification, revision=revision, results=results,
                           registry=registry_path, output=base / 'workflow_v02')


def open_restart(fixture, **kwargs):
    return pipeline.open_workflow(fixture.output, development_root=fixture.results,
                                  development_archive=fixture.archive, fair_q_registry=fixture.registry,
                                  source_revision=fixture.revision, **kwargs)


def test_corrected_workflow_pins_new_root_archive_registry_and_resumes_original_identity(restart):
    identity = open_restart(restart, training_workers=2, evaluation_workers=3)
    assert identity['development_results_root'] == str(restart.results.resolve())
    assert identity['development_source_manifest'] == str(restart.archive.resolve())
    assert identity['development_source_sha256'] == restart.source
    assert identity['fair_q_registry_sha256'] == pipeline.digest(restart.registry)
    assert identity['source_revision_sha256'] == pipeline.digest(restart.revision)
    assert pipeline.open_workflow(restart.output, resume=True) == identity
    assert pipeline.development_status_arguments(identity) == {
        'results_root': str(restart.results.resolve()), 'source_archive': str(restart.archive.resolve())}
    assert not (restart.output / 'logging_migration.json').exists()


@pytest.mark.parametrize('field', ['registry', 'revision', 'archive'])
def test_restart_identity_rejects_changed_bound_input(restart, field):
    open_restart(restart)
    path = getattr(restart, field)
    path.write_bytes(path.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='identity changed|changed'):
        pipeline.open_workflow(restart.output, resume=True)


@pytest.mark.parametrize('field,value', [
    ('old_results_used_for_selection', True),
    ('same_declared_plans_seeds_and_budgets', False),
    ('planned_development_runs', 149),
    ('changed_training_files', []),
])
def test_source_revision_cannot_claim_partial_or_mixed_restart(restart, field, value):
    data = pipeline.read(restart.revision)
    data[field] = value
    restart.revision.write_text(json.dumps(data), encoding='utf-8')
    with pytest.raises(ValueError, match='restart|inventory'):
        open_restart(restart)
    assert not restart.output.exists()


def test_source_revision_rejects_mutated_verification_and_changed_live_source(restart):
    restart.verification.write_text('<testsuite tests="500"/>')
    with pytest.raises(ValueError, match='evidence changed'):
        pipeline.verify_source_revision(restart.revision, restart.source)
    restart.verification.write_text('<testsuite tests="400" failures="0" errors="0" skipped="0"/>')
    restart.model.write_text('# subsequent unverified source change\n')
    with pytest.raises(ValueError, match='source changed'):
        pipeline.verify_source_revision(restart.revision, restart.source)


def test_restart_rejects_other_results_root_before_creating_workflow(restart):
    with pytest.raises(ValueError, match='results root differs'):
        pipeline.open_workflow(restart.output, development_root=restart.base / 'old_development',
                               development_archive=restart.archive, fair_q_registry=restart.registry,
                               source_revision=restart.revision)
    assert not restart.output.exists()


def test_restart_rejects_other_archive_even_if_source_bytes_match(restart):
    other = restart.archive.with_name('unbound_source.json')
    other.write_bytes(restart.archive.read_bytes())
    with pytest.raises(ValueError, match='another archive'):
        pipeline.open_workflow(restart.output, development_root=restart.results,
                               development_archive=other, fair_q_registry=restart.registry,
                               source_revision=restart.revision)
    assert not restart.output.exists()


def test_restart_requires_tests_and_resume_to_be_hash_bound(restart):
    record = pipeline.read(restart.revision)
    record['evidence_files'] = []
    restart.revision.write_text(json.dumps(record), encoding='utf-8')
    with pytest.raises(ValueError, match='hash-bound'):
        open_restart(restart)


@pytest.fixture
def count_setup(tmp_path, monkeypatch):
    root = tmp_path / 'counts'
    root.mkdir()
    core = root / 'core_formal.json'
    pipeline.write_new(core, {'plan_sha256': 'core-selection', 'status': 'frozen'})
    files = {}
    for key in ('export_index', 'physical_check', 'checkpoint_check', 'development_plan'):
        path = root / (key + '.json')
        pipeline.write_new(path, {'synthetic_fixture': key})
        files[key] = {'path': str(path), 'sha256': pipeline.digest(path)}
    index = root / 'count_evidence.json'
    pipeline.write_new(index, {'files': files, 'development_root': str(root / 'development')})
    manifest = {'count_evidence': str(index), 'count_evidence_sha256': pipeline.digest(index),
                'training_workers': 8, 'evaluation_workers': 8}
    calls = []

    def prepare(base_plan, export_index, output):
        assert Path(base_plan) == core
        assert export_index == files['export_index']['path']
        calls.append('prepare')
        pipeline.write_new(output, {'base_protocol': {'path': str(core.resolve())},
                                   'runs': list(range(40)), 'evaluation_jobs': list(range(86)),
                                   'status': 'candidate'})

    def freeze(candidate, physical_check, development_root, checkpoint_check, output):
        assert physical_check == files['physical_check']['path']
        assert checkpoint_check == files['checkpoint_check']['path']
        assert development_root == str(root / 'development')
        calls.append('freeze')
        value = pipeline.read(candidate)
        value.update(status='frozen', plan_sha256='count-plan', evaluation_registry_sha256='count-jobs')
        pipeline.write_new(output, value)

    def validate_count_plan(plan):
        assert plan['status'] == 'frozen' and len(plan['runs']) == 40

    def validate_registry(plan):
        assert len(plan['evaluation_jobs']) == 86

    engine = install_module(monkeypatch, 'experiments.revision_pv_count_pipeline', prepare=prepare, freeze=freeze)
    install_module(monkeypatch, 'experiments.revision_pv_count_plan', validate_count_plan=validate_count_plan)
    queue = install_module(monkeypatch, 'experiments.revision_pv_count_queue', validate_registry=validate_registry)
    return SimpleNamespace(root=root, core=core, files=files, index=index, manifest=manifest, calls=calls,
                           ledger=pipeline.StageLedger(root / 'workflow', 'synthetic-count-workflow'),
                           engine=engine, queue=queue)


def test_count_prepare_uses_current_core_and_bound_evidence_once_on_resume(count_setup):
    fixture = count_setup
    frozen = pipeline.prepare_count_workflow(fixture.core, fixture.manifest, fixture.ledger)
    plan = pipeline.read(frozen['path'])
    assert len(plan['runs']) == 40 and len(plan['evaluation_jobs']) == 86
    assert Path(plan['base_protocol']['path']) == fixture.core
    assert pipeline.prepare_count_workflow(fixture.core, fixture.manifest, fixture.ledger) == frozen
    assert fixture.calls == ['prepare', 'freeze']


@pytest.mark.parametrize('target', ['index', 'physical_check', 'checkpoint_check', 'export_index', 'development_plan'])
def test_count_prepare_rejects_evidence_tamper_before_any_child_or_mutation(count_setup, target):
    fixture = count_setup
    path = fixture.index if target == 'index' else Path(fixture.files[target]['path'])
    path.write_bytes(path.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='changed'):
        pipeline.prepare_count_workflow(fixture.core, fixture.manifest, fixture.ledger)
    assert not fixture.calls


def test_interrupted_count_prepare_preserves_first_candidate_then_uses_new_attempt(count_setup, monkeypatch):
    fixture = count_setup
    original = fixture.engine.freeze

    def interrupted(*args):
        raise KeyboardInterrupt('synthetic parent interruption before frozen count plan')

    monkeypatch.setattr(fixture.engine, 'freeze', interrupted)
    with pytest.raises(KeyboardInterrupt):
        pipeline.prepare_count_workflow(fixture.core, fixture.manifest, fixture.ledger)
    old = fixture.root / 'workflow/stages/count_freeze/a0001/candidate.json'
    before = old.read_bytes()
    monkeypatch.setattr(fixture.engine, 'freeze', original)
    result = pipeline.prepare_count_workflow(fixture.core, fixture.manifest, fixture.ledger)
    assert 'a0002' in result['path'] and old.read_bytes() == before
    assert fixture.calls == ['prepare', 'prepare', 'freeze']


@pytest.mark.parametrize('corrupt_report', [False, True])
def test_count_child_commands_use_40_runs_86_jobs_bounded_workers_and_no_duplicate_resume(count_setup, monkeypatch, corrupt_report):
    fixture = count_setup
    frozen = pipeline.prepare_count_workflow(fixture.core, fixture.manifest, fixture.ledger)
    monkeypatch.setattr(pipeline, 'BASE', fixture.root)
    args = SimpleNamespace(output=fixture.root / 'workflow', count_failure_record=[], retry_failed_evaluations=False)
    state = {'training': False, 'evaluation': False}
    commands = []

    def preflight(plan, root, failures):
        assert len(plan['runs']) == 40 and failures == []
        if not state['training']:
            raise FileNotFoundError('synthetic missing count training')
        return {'completed': list(range(40)), 'reviewed_failures': {}}

    def collect(plan, root):
        if not state['evaluation']:
            raise FileNotFoundError('synthetic missing count evaluation')
        return [{'job': {'job_id': str(i)}, 'status': 'completed',
                 'evaluation_sha256': 'evaluation-' + str(i)} for i in range(86)]

    fixture.queue.preflight_training = preflight
    fixture.queue.collect_verified_results = collect

    def run_command(command, log_path):
        commands.append(command)
        if 'train' in command:
            assert command[command.index('--max-runs') + 1] == '40'
            assert command[command.index('--workers') + 1] == '4'
            Path(command[command.index('--training-root') + 1]).mkdir(parents=True)
            state['training'] = True
        elif 'dispatch' in command:
            assert command[command.index('--workers') + 1] == '4'
            queue_root = Path(command[command.index('--output') + 1])
            pipeline.write_new(queue_root / 'queue_identity.json', {'fixture': True})
            state['evaluation'] = True
        else:
            assert any('revision_pv_count_reporting.py' in part for part in command)
            pipeline.write_new(Path(command[command.index('--output') + 1]), {
                'plan_sha256': 'count-plan', 'evaluation_registry_sha256': 'count-jobs',
                'planned_evaluation_jobs': 85 if corrupt_report else 86,
                'groups': [{'n_planned_training_seeds': 10} for _ in range(8)],
                'traditional': [{} for _ in range(6)],
            })

    monkeypatch.setattr(pipeline, 'run_command', run_command)
    if corrupt_report:
        with pytest.raises(ValueError, match='complete registered'):
            pipeline.run_count_workflow(frozen, fixture.manifest, fixture.ledger, args)
    else:
        report = pipeline.run_count_workflow(frozen, fixture.manifest, fixture.ledger, args)
        assert pipeline.run_count_workflow(frozen, fixture.manifest, fixture.ledger, args) == report
    assert len(commands) == 3
    assert not any('--resume' in command for command in commands)
