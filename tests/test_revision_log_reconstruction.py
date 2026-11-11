import pytest
from manuscript_revision.revision_training_logs import canonical_reward


def test_corrects_duplicate_component_without_halving_failure_penalty():
    result = canonical_reward({'mean_train_total_loss': .2, 'mean_train_destroy': .05,
                               'mean_train_reward': -10.4}, source_confirmed=True)
    assert result['canonical_mean_train_reward'] == pytest.approx(-10.2)
    assert result['logging_status'] == 'duplicated_info_reward_confirmed'


def test_already_fixed_logs_are_never_blindly_divided():
    result = canonical_reward({'mean_train_total_loss': .2, 'mean_train_destroy': .05,
                               'mean_train_reward': -10.2}, source_confirmed=True)
    assert result['canonical_mean_train_reward'] == pytest.approx(-10.2)
    assert result['logging_status'] == 'already_single_counted'


def test_unverified_source_or_missing_destroy_is_rejected():
    with pytest.raises(ValueError, match='audited source'):
        canonical_reward({})
    with pytest.raises(ValueError, match='requires loss'):
        canonical_reward({'mean_train_total_loss': .2, 'mean_train_reward': -.4}, source_confirmed=True)
