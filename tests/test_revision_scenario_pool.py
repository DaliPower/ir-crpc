import numpy as np
import pytest

from experiments.revision_scenario_pool import ScenarioSampler, classify_risk, validate_pool


def pool():
    return [{'day': day, 'pv_scale': k, 'stratum': 'over' if day == 1 else 'low'}
            for day in range(1, 5) for k in (.5, 1., 1.5)]


def test_no_future_or_incomplete_day_in_pool():
    for day in (0, 999, 1000, 1030):
        rows = pool()
        rows[0]['day'] = day
        with pytest.raises(ValueError):
            validate_pool(rows)


def test_pool_requires_same_kappa_support_per_day():
    with pytest.raises(ValueError, match='three scales'):
        validate_pool(pool()[:-1])


def test_b0_fixed_scale_and_b1_uniform_common_support():
    b0 = ScenarioSampler(pool(), 'B0', 9)
    b1 = ScenarioSampler(pool(), 'B1', 9)
    assert {b0.sample()['pv_scale'] for _ in range(100)} == {1.5}
    np.testing.assert_allclose(b1.probability, np.ones(12) / 12)


def test_b2_mixture_retains_low_risk_and_balances_strata():
    sampler = ScenarioSampler(pool(), 'B2', 9, mixture=.5)
    assert np.all(sampler.probability > 0)
    assert np.isclose(sampler.probability[:3].sum(), .375)
    assert np.isclose(sampler.probability.sum(), 1)


def test_resume_reproduces_next_scenarios_and_exposures():
    first = ScenarioSampler(pool(), 'B2', 1)
    for _ in range(17):
        first.sample()
    second = ScenarioSampler(pool(), 'B2', 2)
    second.load_state_dict(first.state_dict())
    assert [first.sample() for _ in range(40)] == [second.sample() for _ in range(40)]
    np.testing.assert_array_equal(first.exposures, second.exposures)


def test_resume_rejects_different_sampling_protocol():
    first = ScenarioSampler(pool(), 'B2', 1)
    second = ScenarioSampler(pool(), 'B1', 1)
    with pytest.raises(ValueError, match='protocol mismatch'):
        second.load_state_dict(first.state_dict())


@pytest.mark.parametrize('hi,lo,fail,expected', [(0,0,0,'low'),(.1,0,0,'over'),
    (0,.1,0,'under'),(.1,.1,0,'mixed'),(0,0,1,'powerflow_failure')])
def test_explicit_risk_categories(hi, lo, fail, expected):
    assert classify_risk(hi, lo, fail) == expected
