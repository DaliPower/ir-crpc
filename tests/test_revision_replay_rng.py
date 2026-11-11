import numpy as np
import pytest

from utilities.replay_buffer import EpisodeReplayBuffer, TransReplayBuffer


@pytest.mark.parametrize('buffer_class', [TransReplayBuffer, EpisodeReplayBuffer])
def test_replay_sampling_is_reproducible_and_independent_of_global_rng(buffer_class):
    left, right = buffer_class(20, seed=271), buffer_class(20, seed=271)
    for i in range(20):
        item = [i] if buffer_class is EpisodeReplayBuffer else i
        left.add_experience(item)
        right.add_experience(item)
    for _ in range(8):
        expected = left.get_batch(4)
        np.random.seed(123)
        np.random.normal(size=1000)
        assert right.get_batch(4) == expected


@pytest.mark.parametrize('buffer_class', [TransReplayBuffer, EpisodeReplayBuffer])
def test_replay_does_not_advance_global_numpy_state(buffer_class):
    np.random.seed(91)
    expected = np.random.random(5)
    np.random.seed(91)
    buffer = buffer_class(20, seed=10)
    for i in range(10):
        buffer.add_experience([i] if buffer_class is EpisodeReplayBuffer else i)
    buffer.get_batch(4)
    np.testing.assert_array_equal(np.random.random(5), expected)
