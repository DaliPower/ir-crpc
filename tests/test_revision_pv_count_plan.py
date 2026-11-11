import copy
import json
from pathlib import Path
import pytest

from experiments.revision_run_plan import build_plan, SCENARIOS
from experiments.revision_pv_count_plan import (build_count_plan, validate_count_structure,
    validate_count_plan, count_source_manifest, freeze_count_plan, ROOT)


def candidate(tmp_path, phase='formal'):
    for scenario in SCENARIOS:
        (tmp_path/(scenario+'_pool.json')).write_text('{}')
    base = tmp_path/'base.json'
    base.write_text(json.dumps(build_plan(tmp_path)))
    index = tmp_path/'export_index.json'
    index.write_text(json.dumps({s:{'dataset_manifest':str(tmp_path/(s+'_pool.json')),
                                   'pool':str(tmp_path/(s+'_pool.json'))} for s in SCENARIOS}))
    return build_count_plan(base, index, phase=phase)


def test_count_protocol_is_separate_40_run_full_budget_matrix(tmp_path):
    plan = candidate(tmp_path)
    validate_count_structure(plan)
    assert len(plan['runs']) == 40
    assert {r['solver'] for r in plan['runs']} == {'matd3_reference','maddpg'}
    assert {r['seed'] for r in plan['runs']} == set(range(10))
    assert plan['protocol']['training_steps'] == 240000
    assert plan['protocol']['validation_days'] == list(range(1000,1030))
    assert plan['protocol']['formal_test_days'] == list(range(1030,1060))
    assert plan['base_protocol']['selection_is_provisional']


@pytest.mark.parametrize('edit', ['budget','dates','duplicate','missing_seed','reward'])
def test_count_formal_structure_rejects_protocol_shortcuts(tmp_path, edit):
    plan = candidate(tmp_path)
    if edit == 'budget':
        plan['protocol']['training_steps'] = 480
    elif edit == 'dates':
        plan['protocol']['validation_days'] = [1000]
    elif edit == 'duplicate':
        plan['runs'][-1] = copy.deepcopy(plan['runs'][0])
    elif edit == 'missing_seed':
        plan['runs'].pop()
    else:
        plan['protocol']['reward']['q_weight'] = .3
    with pytest.raises(ValueError):
        validate_count_structure(plan)


def test_count_smoke_is_explicitly_disjoint_two_run_development(tmp_path):
    plan = candidate(tmp_path, 'development')
    validate_count_structure(plan)
    assert len(plan['runs']) == 2
    assert all(r['seed'] == 1901 for r in plan['runs'])
    assert plan['protocol']['validation_days'] == [1000]
    assert plan['status'] == 'development_only'


def test_count_formal_cannot_freeze_before_core_selection_is_frozen(tmp_path):
    with pytest.raises(ValueError,match='candidate protocol'):
        freeze_count_plan(candidate(tmp_path),{})


def test_count_source_locks_count_dependencies_without_unrelated_reporting():
    sources = count_source_manifest()['source_sha256']
    for name in ('revision_pv_count_environment.py','revision_pv_count_export.py',
                 'revision_pv_count_plan.py','run_revision_pv_count_training.py'):
        assert 'experiments/'+name in sources
    assert 'experiments/revision_benchmark.py' not in sources
    assert 'experiments/revision_evaluation.py' not in sources


def test_controlled_copy_has_only_declared_entry_point_changes():
    evidence = ROOT/'docs/revision_access_41572/evidence/count_runner_copy'
    original = (evidence/'core_runner_at_copy.py.txt').read_text(encoding='utf-8')
    assert original == (ROOT/'experiments/run_revision_training.py').read_text(encoding='utf-8'), 'Re-audit copy after any core change'
    for change in json.loads((evidence/'controlled_changes.json').read_text(encoding='utf-8'))['changes']:
        assert original.count(change['old']) == 1
        original = original.replace(change['old'],change['new'])
    assert original == (ROOT/'experiments/run_revision_pv_count_training.py').read_text(encoding='utf-8')
