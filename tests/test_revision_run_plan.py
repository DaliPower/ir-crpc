import copy
import json
from pathlib import Path
import pytest

from experiments.revision_run_plan import (build_plan, freeze_plan, canonical_hash,
                                          validate_frozen_plan)


def pool_dir(tmp_path):
    for scenario in ('case33_3min_final','case141_3min_final'):
        data = tmp_path / scenario
        data.mkdir(exist_ok=True)
        (data / 'model.p').write_text('fixture')
        from experiments.run_revision_pilot import file_hash
        (tmp_path / (scenario+'_pool.json')).write_text(json.dumps({
            'rows': [{'day': day, 'pv_scale': scale, 'stratum': 'low'}
                     for day in range(1, 999) for scale in (.5, 1., 1.5)],
            'data_directory': str(data), 'data_sha256': {'model.p': file_hash(data / 'model.p')}}), encoding='utf-8')
    return tmp_path


def test_formal_matrix_covers_ten_seeds_eight_variants_two_networks(tmp_path):
    plan = build_plan(pool_dir(tmp_path))
    assert len(plan['runs']) == 160
    assert len({r['run_id'] for r in plan['runs']}) == 160
    assert plan['protocol']['training_steps'] == 240000
    assert plan['protocol']['episode_points'] == 480
    assert set(r['seed'] for r in plan['runs']) == set(range(10))
    assert plan['protocol']['validation_days'] == list(range(1000,1030))
    with pytest.raises(ValueError, match='candidate'):
        validate_frozen_plan(plan, verify_source=False)


@pytest.mark.parametrize('kw', [dict(seeds=[0,1,2]),dict(seeds=list(range(100,110))),
    dict(validation_days=[1030]),dict(phase='development',seeds=[0])])
def test_no_seed_or_test_date_shortcuts(tmp_path, kw):
    with pytest.raises(ValueError):
        build_plan(pool_dir(tmp_path), **kw)


def test_cannot_freeze_without_completed_development_gate(tmp_path):
    with pytest.raises(ValueError, match='gate incomplete'):
        freeze_plan(build_plan(pool_dir(tmp_path)), {'implementation_tests_passed':True})


def gate(tmp_path):
    from experiments.run_revision_pilot import file_hash
    evidence = tmp_path/'check.json'
    evidence.write_text('{"status":"passed"}')
    values = {k:True for k in ('implementation_tests_passed','all_five_solvers_exercised',
        'resume_verified','traditional_protocol_aligned','validation_diagnosis_completed',
        'configuration_selected_on_validation','runtime_budget_measured')}
    values['evidence_files'] = [{'path':str(evidence),'sha256':file_hash(evidence)}]
    values['assertion_evidence'] = {key: [str(evidence)] for key in values if key != 'evidence_files'}
    values['development_runs'] = []
    for scenario in ('case33_3min_final', 'case141_3min_final'):
        for solver in ('matd3_reference', 'matd3', 'maddpg', 'mappo', 'facmaddpg'):
            path = tmp_path / f'{scenario}_{solver}.json'
            path.write_text(json.dumps({'status': 'completed', 'phase': 'development',
                'environment_steps': 480, 'wall_seconds': 1., 'formal_test_evaluated': False,
                'run': {'scenario': scenario, 'solver': solver, 'seed': 100}}))
            values['development_runs'].append({'path': str(path)})
            values['evidence_files'].append({'path': str(path), 'sha256': file_hash(path)})
    return values


def test_frozen_plan_detects_changed_budget_and_pool(tmp_path):
    plan = freeze_plan(build_plan(pool_dir(tmp_path)), gate(tmp_path), source={'source_sha256':{}})
    validate_frozen_plan(plan, verify_source=False)
    bad = copy.deepcopy(plan)
    bad['protocol']['training_steps'] = 400
    with pytest.raises(ValueError, match='was changed'):
        validate_frozen_plan(bad, verify_source=False)
    Path(plan['pools']['case33_3min_final']['path']).write_text('tampered')
    with pytest.raises(ValueError, match='pool changed'):
        validate_frozen_plan(plan, verify_source=False)


def test_freeze_checks_actual_evidence_hash(tmp_path):
    values = gate(tmp_path)
    values['evidence_files'][0]['sha256'] = 'not-a-real-checksum'
    with pytest.raises(ValueError, match='evidence file hash'):
        freeze_plan(build_plan(pool_dir(tmp_path)), values)


@pytest.mark.parametrize('change',['missing','changed'])
def test_formal_plan_requires_the_explicit_supported_validation_rule(tmp_path,change):
    from experiments.revision_run_plan import validate_protocol_structure
    from experiments.revision_validation_selection import VALIDATION_SELECTION_POLICY
    plan=build_plan(pool_dir(tmp_path))
    assert plan['protocol']['validation_selection_policy']==VALIDATION_SELECTION_POLICY
    if change=='missing': plan['protocol'].pop('validation_selection_policy')
    else: plan['protocol']['validation_selection_policy']['unexpected_nonfinite']='accept'
    with pytest.raises(ValueError,match='validation selection policy'):
        validate_protocol_structure(plan)
    assert VALIDATION_SELECTION_POLICY['unexpected_nonfinite'].startswith('reject')
