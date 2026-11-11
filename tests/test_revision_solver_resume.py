"""Small deterministic mechanism checks, not experimental performance evidence."""
from copy import deepcopy
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from experiments.revision_checkpointing import (
    save_training_checkpoint, load_training_checkpoint, preserve_training_random_state)
from models.model_registry import Model
from utilities.trainer import PGTrainer
from utilities.util import squashed_normal_log_density

ROOT = Path(__file__).resolve().parents[1]


def arguments(solver='maddpg', **overrides):
    values = yaml.safe_load((ROOT / 'args/default.yaml').read_text())
    values.update(yaml.safe_load((ROOT / f'args/alg_args/{solver}.yaml').read_text())['alg_args'])
    values.update(cuda=False, seed=37, agent_num=2, obs_size=3, action_dim=1,
                  hid_size=8, batch_size=2, layernorm=False, action_bias=0.0, action_scale=0.6,
                  reward_normalisation=False, normalize_advantages=False, entr=0.0,
                  replay_seed=31, exploration_seed=33, target_noise_seed=35,
                  max_steps=3, behaviour_update_freq=1, value_update_epochs=1,
                  policy_update_epochs=1, target_update_freq=2, replay_warmup=0)
    values.update(overrides)
    return SimpleNamespace(**values)


class TinyEnvironment:
    """A stochastic, cheap reset boundary with a custom scenario sampler."""
    def __init__(self):
        self.rng = np.random.default_rng(19)
        self.sampler = np.random.default_rng(23)
        self.exposure = []
        self.count = 0

    def checkpoint_state_dict(self):
        return {'sampler_rng': deepcopy(self.sampler.bit_generator.state),
                'exposure': self.exposure.copy()}

    def load_checkpoint_state_dict(self, state):
        self.sampler.bit_generator.state = state['sampler_rng']
        self.exposure = state['exposure'].copy()

    def reset(self):
        self.count = 0
        self.exposure.append(int(self.sampler.integers(1000)))
        self.observation = self.rng.normal(size=(2, 3)).astype(np.float32)
        return self.observation.copy(), self.observation.flatten()

    def step(self, action):
        self.count += 1
        self.observation = self.rng.normal(size=(2, 3)).astype(np.float32)
        return -float(np.square(action).sum()), self.count == 3, {'destroy': 0.0}

    def get_obs(self):
        return self.observation.copy()

    def get_avail_actions(self):
        return np.ones((1, 2, 1), dtype=np.float32)

    def get_num_of_agents(self):
        return 2


def trainer_for(solver='maddpg', **overrides):
    return PGTrainer(arguments(solver, **overrides), Model[solver],
                     TinyEnvironment(), SimpleNamespace(add_scalar=lambda *args: None))


def assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_nested_equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize('solver', ['maddpg', 'facmaddpg', 'mappo', 'matd3', 'matd3_reference'])
@pytest.mark.parametrize('cuda', [False, pytest.param(True, marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_resumed_next_episode_samples_and_parameter_updates_equal_uninterrupted(tmp_path, solver, cuda):
    torch.manual_seed(101)
    random.seed(103)
    np.random.seed(107)
    original = trainer_for(solver, cuda=cuda)
    original.behaviour_net.train_process({}, original)
    checkpoint = save_training_checkpoint(tmp_path / 'state.pt', original,
        progress={'next_evaluation': 12}, metadata={'source': 'test-fixture'})
    original.behaviour_net.train_process({}, original)
    expected_state = deepcopy(original.behaviour_net.state_dict())
    expected_optimizer = deepcopy(original.policy_optimizer.state_dict())
    expected_value_optimizer = deepcopy(original.value_optimizer.state_dict())
    expected_global_draws = (random.random(), np.random.random(), torch.rand(4),
                             torch.rand(4, device=original.device))
    expected_replay = deepcopy(original.replay_buffer.buffer)
    restored = trainer_for(solver, cuda=cuda)
    progress = load_training_checkpoint(checkpoint, restored,
                                       expected_metadata={'source': 'test-fixture'})
    assert progress == {'next_evaluation': 12}
    restored.behaviour_net.train_process({}, restored)
    assert_nested_equal(expected_state, restored.behaviour_net.state_dict())
    assert_nested_equal(expected_optimizer, restored.policy_optimizer.state_dict())
    assert_nested_equal(expected_value_optimizer, restored.value_optimizer.state_dict())
    assert_nested_equal(expected_global_draws,
                        (random.random(), np.random.random(), torch.rand(4),
                         torch.rand(4, device=restored.device)))
    assert_nested_equal(expected_replay, restored.replay_buffer.buffer)
    assert restored.steps == original.steps == 6
    assert restored.episodes == original.episodes == 2
    assert restored.update_counts == original.update_counts
    assert restored.env.exposure == original.env.exposure
    if solver == 'facmaddpg':
        assert restored.update_counts['mixer'] > 0
        assert_nested_equal(original.mixer_optimizer.state_dict(), restored.mixer_optimizer.state_dict())


@pytest.mark.parametrize('solver', ['maddpg', 'facmaddpg', 'mappo'])
def test_baseline_exploration_is_local_and_evaluation_does_not_advance_it(solver):
    trainer = trainer_for(solver)
    model = trainer.behaviour_net
    obs = torch.zeros(1, 2, 3)
    hidden = torch.zeros(1, 2, 8)
    mask = torch.ones(1, 2, 1)
    local = model.exploration_generator.get_state().clone()
    global_before = torch.random.get_rng_state().clone()
    model.get_actions(obs, 'train', True, mask, last_hid=hidden)
    assert torch.equal(global_before, torch.random.get_rng_state())
    assert not torch.equal(local, model.exploration_generator.get_state())
    local = model.exploration_generator.get_state().clone()
    model.get_actions(obs, 'test', False, mask, last_hid=hidden)
    assert torch.equal(local, model.exploration_generator.get_state())
    assert model.target_noise_generator is None  # no stochastic target smoothing


@pytest.mark.parametrize('solver', ['maddpg', 'facmaddpg', 'mappo'])
def test_explicit_legacy_rng_uses_global_stream(solver):
    model = trainer_for(solver, solver_rng_mode='global_legacy').behaviour_net
    before = torch.random.get_rng_state().clone()
    model.get_actions(torch.zeros(1, 2, 3), 'train', True, torch.ones(1, 2, 1),
                      last_hid=torch.zeros(1, 2, 8))
    assert model.exploration_generator is None
    assert not torch.equal(before, torch.random.get_rng_state())


@pytest.mark.parametrize('action_dim', [1, 2])
def test_mappo_stored_log_probability_matches_new_density_before_any_update(action_dim):
    model = trainer_for('mappo', action_dim=action_dim).behaviour_net
    obs = torch.full((1, 2, 3), 0.2)
    hidden = torch.zeros(1, 2, 8)
    mask = torch.ones(1, 2, 1) if action_dim == 1 else torch.eye(2).unsqueeze(0)
    action, restored, old_log, _, _ = model.get_actions(obs, 'train', True, mask, last_hid=hidden)
    means, logs, _ = model.policy(obs, last_hid=hidden)
    means, logs = model.masked_distribution(means, logs, mask)
    new_log = squashed_normal_log_density(restored, means, logs)
    ratio = ((new_log * mask).sum(-1) - (old_log * mask).sum(-1)).exp()
    assert torch.allclose(ratio, torch.ones_like(ratio), atol=1e-6)
    assert action.abs().max() <= 1
    assert not model.args.target
    assert not hasattr(model, 'target_net')


def test_unpack_uses_log_prob_field_not_action():
    trainer = trainer_for('mappo', replay_warmup=999)
    trainer.behaviour_net.train_process({}, trainer)
    batch = trainer.behaviour_net.Transition(*zip(*trainer.replay_buffer.buffer))
    data = trainer.behaviour_net.unpack_data(batch)
    np.testing.assert_array_equal(data[2].numpy(), np.concatenate(batch.log_prob_a))
    assert not torch.equal(data[1], data[2])


def test_budget_truncation_and_terminal_skip_next_measurement():
    trainer = trainer_for(replay_warmup=999)
    trainer.behaviour_net.train_process({}, trainer, step_limit=2)
    assert trainer.steps == 2
    assert trainer.last_episode_reason == 'budget_truncation'
    assert trainer.replay_buffer.buffer[-1].last_step
    assert not trainer.replay_buffer.buffer[-1].done
    trainer = trainer_for(replay_warmup=999)
    real_get = trainer.env.get_obs
    def guarded_obs():
        assert trainer.env.count < 3, 'must not query terminal observation'
        return real_get()
    trainer.env.get_obs = guarded_obs
    trainer.behaviour_net.train_process({}, trainer)
    assert trainer.steps == 3 and trainer.replay_buffer.buffer[-1].done


def test_precontrol_failure_is_explicit_terminal_but_programming_error_propagates():
    from pandapower.powerflow import LoadflowNotConverged
    trainer = trainer_for(replay_warmup=999)
    def fail():
        raise LoadflowNotConverged('fixture')
    trainer.env.get_obs = fail
    stat = {}
    trainer.behaviour_net.train_process(stat, trainer)
    assert trainer.last_episode_reason == 'pre_control_power_flow_failure'
    assert trainer.steps == 1 and trainer.replay_buffer.buffer[-1].done
    assert stat['mean_train_pre_control_pf_failed'] == 1
    trainer.env.get_obs = lambda: (_ for _ in ()).throw(RuntimeError('program bug'))
    with pytest.raises(RuntimeError, match='program bug'):
        trainer.behaviour_net.train_process({}, trainer)
    assert not trainer.at_episode_boundary


def test_checkpoint_rejects_inflight_and_mismatched_metadata(tmp_path):
    trainer = trainer_for()
    trainer.at_episode_boundary = False
    with pytest.raises(ValueError, match='episode boundary'):
        save_training_checkpoint(tmp_path / 'invalid.pt', trainer, progress={})
    trainer.at_episode_boundary = True
    path = save_training_checkpoint(tmp_path / 'valid.pt', trainer, progress={}, metadata={'source': 'one'})
    with pytest.raises(ValueError, match='metadata mismatch'):
        load_training_checkpoint(path, trainer, expected_metadata={'source': 'two'})


def test_evaluation_context_restores_model_modes_auxiliary_stats_and_all_streams():
    trainer = trainer_for('mappo')
    model = trainer.behaviour_net
    original_global = torch.random.get_rng_state().clone()
    original_local = model.exploration_generator.get_state().clone()
    auxiliary_state = deepcopy(model.rl.batchnorm.state_dict())
    with preserve_training_random_state(trainer):
        model.eval()
        model.rl.batchnorm(torch.randn(4, 2))
        model.exploration_generator.manual_seed(999)
        trainer.env.sampler.integers(1000)
    assert model.training
    assert torch.equal(original_global, torch.random.get_rng_state())
    assert torch.equal(original_local, model.exploration_generator.get_state())
    assert_nested_equal(auxiliary_state, model.rl.batchnorm.state_dict())


def test_checkpoint_restores_ppo_auxiliary_normalization_buffers(tmp_path):
    trainer = trainer_for('mappo', normalize_advantages=True)
    trainer.behaviour_net.train_process({}, trainer)
    expected = deepcopy(trainer.behaviour_net.rl.batchnorm.state_dict())
    path = save_training_checkpoint(tmp_path / 'ppo.pt', trainer, progress={})
    restored = trainer_for('mappo', normalize_advantages=True)
    load_training_checkpoint(path, restored)
    assert_nested_equal(expected, restored.behaviour_net.rl.batchnorm.state_dict())


def test_checkpoint_preserves_episode_replay_and_next_random_batch(tmp_path):
    source = trainer_for(replay_warmup=999)
    source.behaviour_net.train_process({}, source)
    trainer = trainer_for(episodic=True)
    episode = source.replay_buffer.buffer
    for index in range(4):
        trainer.replay_buffer.add_experience([row._replace(reward=np.full(2, index)) for row in episode])
    path = save_training_checkpoint(tmp_path / 'episode.pt', trainer, progress={'boundary': 0})
    expected = trainer.replay_buffer.get_batch(2)
    restored = trainer_for(episodic=True)
    load_training_checkpoint(path, restored)
    assert_nested_equal(expected, restored.replay_buffer.get_batch(2))


def test_mappo_gae_does_not_cross_a_truncated_episode_boundary():
    trainer = trainer_for('mappo', replay_warmup=999, gamma=0.5, lambda_=1.0)
    trainer.behaviour_net.train_process({}, trainer)
    model = trainer.behaviour_net
    rows = trainer.replay_buffer.buffer[:2]
    rows = [row._replace(reward=np.full(2, reward), value=np.zeros((1, 2, 1)),
                         next_value=np.zeros((1, 2, 1)), done=False, last_step=last)
            for row, reward, last in zip(rows, [1.0, 100.0], [True, False])]
    batch = model.Transition(*zip(*rows))
    loss, _, _ = model.get_loss(batch)
    # Both current/old densities match. First episode's advantage is 1 rather
    # than 1 + gamma * 100 from the next episode after its truncation boundary.
    assert loss.item() == pytest.approx(-50.5, abs=1e-5)


def test_evaluation_context_restores_replay_rng_and_registered_running_buffers():
    trainer = trainer_for(replay_warmup=999)
    trainer.behaviour_net.train_process({}, trainer)
    replay_state = deepcopy(trainer.replay_buffer.rng.bit_generator.state)
    buffers = deepcopy(trainer.behaviour_net.batchnorm.state_dict())
    with preserve_training_random_state(trainer):
        trainer.replay_buffer.get_batch(2)
        trainer.behaviour_net.batchnorm(torch.randn(4, 2))
    assert_nested_equal(replay_state, trainer.replay_buffer.rng.bit_generator.state)
    assert_nested_equal(buffers, trainer.behaviour_net.batchnorm.state_dict())
