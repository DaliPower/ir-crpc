import numpy as np
import pytest
from experiments.revision_statistics import (seed_summary,independent_comparison,
                                            holm_adjust,validate_seed_records)


def test_seed_variance_is_sample_variance_and_degenerate_interval_is_disclosed():
    result=seed_summary([1,2,3],resamples=100)
    assert result['sample_std']==1
    assert seed_summary([0]*10,resamples=100)['degenerate_interval_does_not_establish_zero_risk']


def test_exact_independent_permutation_known_small_example():
    result=independent_comparison([0,0],[1,1])
    assert result['permutations']==6
    assert result['p_two_sided']==pytest.approx(2/6)
    assert result['difference_first_minus_second']==-1


def test_holm_ordering_and_monotonicity():
    np.testing.assert_allclose(holm_adjust([.04,.01,.03]),[.06,.03,.06])


def test_dates_cannot_replace_ten_training_seeds():
    rows=[{'seed':0,'planned_days':30,'planned_steps':14400,'vrate':.1} for _ in range(10)]
    with pytest.raises(ValueError,match='independent seeds'):
        validate_seed_records(rows)


def test_incomplete_seed_set_or_shortened_day_denominator_rejected():
    rows=[{'seed':s,'planned_days':30,'planned_steps':14400,'vrate':.1} for s in range(10)]
    assert len(validate_seed_records(rows))==10
    rows[3]['planned_steps']=100
    with pytest.raises(ValueError,match='denominators'):
        validate_seed_records(rows)
