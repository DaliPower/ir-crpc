from pathlib import Path
import random

import numpy as np
import pytest
import torch

from experiments.run_revision_pilot import (
    derive_seed_streams, environment_config, isolated_validation_rng,
    validate_pilot_request, validate_output_root,
)


def test_seed_streams_are_named_distinct_and_reproducible():
    streams = derive_seed_streams(100)
    assert streams == derive_seed_streams(100)
    assert len(set(streams.values())) == len(streams)
    assert streams != derive_seed_streams(101)
    assert {'initialization', 'learning', 'training_environment', 'validation_environment',
            'replay', 'exploration', 'target_noise'} <= set(streams)


def test_revision_environment_locks_training_statistics_and_nameplate():
    train = environment_config('case141_3min_final', 100, phase='training')
    validation = environment_config('case141_3min_final', 100, phase='validation')
    assert train['statistics_end_day_exclusive'] == validation['statistics_end_day_exclusive'] == 999
    assert train['capacity_reference_pv_scale'] == validation['capacity_reference_pv_scale'] == 1.5
    assert train['strict_episode_timestamps'] is True
    assert train['start_hour'] == validation['start_hour'] == 6
    assert set(train['start_day_candidates']).isdisjoint(validation['start_day_candidates'])
    assert max(train['start_day_candidates']) < min(validation['start_day_candidates'])
    assert train['seed'] != validation['seed']
    assert train['reset_action'] is validation['reset_action'] is False


@pytest.mark.parametrize('days', [[999], [1030], [1000, 1031], []])
def test_pilot_cannot_consume_final_test_or_outside_validation_dates(days):
    with pytest.raises(ValueError, match='validation'):
        validate_pilot_request(episodes=4, validation_days=days, steps=240)


def test_pilot_cannot_be_mislabeled_as_formal_run(tmp_path):
    with pytest.raises(ValueError, match='pilot'):
        validate_output_root(tmp_path / 'formal', repository=tmp_path)
    accepted = tmp_path / 'exp' / 'revision_access_41572' / 'pilot' / 'new'
    assert validate_output_root(accepted, repository=tmp_path) == accepted.resolve()
    accepted.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        validate_output_root(accepted, repository=tmp_path)


def test_trajectory_writer_allows_separate_revision_root_and_still_refuses_overwrite(tmp_path):
    from experiments.fixed_test_protocol import _validate_output_dir
    destination = tmp_path / 'exp' / 'revision_access_41572' / 'pilot' / 'validation_best'
    _validate_output_dir(destination)
    destination.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        _validate_output_dir(destination)
    with pytest.raises(ValueError):
        _validate_output_dir(tmp_path / 'unregistered_old_results')


def test_validation_preserves_global_training_random_states_even_on_error():
    random.seed(12)
    np.random.seed(13)
    torch.manual_seed(14)
    states = random.getstate(), np.random.get_state(), torch.get_rng_state()
    with pytest.raises(RuntimeError):
        with isolated_validation_rng():
            random.random()
            np.random.random()
            torch.rand(10)
            raise RuntimeError('validation failure')
    assert random.getstate() == states[0]
    assert np.random.get_state()[0] == states[1][0]
    np.testing.assert_array_equal(np.random.get_state()[1], states[1][1])
    assert torch.equal(torch.get_rng_state(), states[2])
