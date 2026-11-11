import copy
import pytest
from experiments.round2_ablation import build_plan, validate_plan, plan_hash, seed_streams, freeze


def test_formal_matrix_and_disjoint_random_domains():
    plan = build_plan('formal')
    assert len(plan['runs']) == 120
    assert plan['protocol']['training_steps'] == 240000
    assert plan['protocol']['formal_test_days'] == list(range(1030,1060))
    assert {r['round2_variant'] for r in plan['runs']} == {f'A{i}' for i in range(6)}
    rng = [seed_streams(r['variant'],r['scenario'],r['seed']) for r in plan['runs']]
    assert len({x['domain_sha256'] for x in rng}) == 120


def test_development_is_validation_only_and_not_formal_relabelable():
    p = build_plan('development',smoke_steps=120)
    assert len(p['runs']) == 12
    assert p['protocol']['validation_days'] == [1000]
    assert all(r['seed'] >= 100 for r in p['runs'])
    p['protocol']['phase'] = 'formal'
    p['status'] = 'frozen'; p['plan_sha256'] = plan_hash(p)
    with pytest.raises(ValueError,match='budget|matrix|source|gate'):
        validate_plan(p,verify_source=False)


def test_frozen_hash_rejects_mutation_before_work():
    p = build_plan('development',smoke_steps=120)
    p['status'] = 'frozen'; p['plan_sha256'] = plan_hash(p)
    p['runs'][0]['seed'] = 9
    with pytest.raises(ValueError,match='hash'):
        validate_plan(p,verify_source=False)


def test_development_freeze_validates_existing_string_pool_paths():
    p=freeze(build_plan('development',smoke_steps=120))
    validate_plan(p)
    assert p['source']['source_sha256']['models/matd3_round2_ablation.py']
