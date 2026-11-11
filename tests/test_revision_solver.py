"""Behavioral solver gates; tiny synthetic batches, never power-grid training."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch as th
import yaml

from models.matd3 import MATD3
from utilities.util import select_action


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def deterministic_torch_fixture():
    with th.random.fork_rng(devices=[]):
        th.manual_seed(42)
        yield


def arguments(**overrides):
    values = yaml.safe_load((ROOT / 'args/default.yaml').read_text())
    values.update(yaml.safe_load((ROOT / 'args/alg_args/matd3.yaml').read_text())['alg_args'])
    values.update(cuda=False, agent_num=2, obs_size=3, action_dim=1, hid_size=8,
                  batch_size=2, layernorm=False, reward_normalisation=False,
                  entr=0.0, policy_delay=2, target_policy_noise=0.2,
                  target_noise_clip=0.5, exploration_noise=0.1,
                  replay_seed=123, behaviour_update_freq=1, value_update_epochs=1)
    values.update(overrides)
    return SimpleNamespace(**values)


def constant_actor(model, output):
    with th.no_grad():
        for actor in model.policy_dicts:
            for parameter in actor.parameters():
                parameter.zero_()
            layer = actor.fc2 if model.args.agent_type == 'rnn' else actor.fc3
            layer.bias.fill_(output)


def batch_for(model):
    a = model.args
    rows = []
    for index in range(a.batch_size):
        rows.append(model.Transition(
            np.full((a.agent_num, a.obs_size), index * 0.1, dtype=np.float32),
            np.full((1, a.agent_num, a.action_dim), 0.2, dtype=np.float32),
            np.zeros((1, a.agent_num, a.action_dim)), np.zeros((2, a.agent_num, 1)),
            np.zeros((2, a.agent_num, 1)), np.full(a.agent_num, 1.0),
            np.full((a.agent_num, a.obs_size), 0.3, dtype=np.float32),
            False, False, np.ones((1, a.agent_num, a.action_dim)),
            np.zeros((1, a.agent_num, a.hid_size)),
            np.zeros((1, a.agent_num, a.hid_size))))
    return model.Transition(*zip(*rows))


def test_actor_training_uses_same_bounded_action_and_tanh_gradient_as_execution():
    args = arguments()
    raw = th.tensor([2.0], requires_grad=True)
    train, _ = select_action(args, raw, exploration=False, info={'log_std': th.zeros(1)})
    executed, _ = select_action(args, raw, status='test', info={'log_std': th.zeros(1)})
    assert train.item() == pytest.approx(0.96402758)
    assert train.item() == executed.item()
    train.sum().backward()
    assert raw.grad.item() == pytest.approx(0.07065082)


def test_matd3_actor_loss_supplies_bounded_actions_to_real_critic():
    args = arguments()
    model = MATD3(args, MATD3(args))
    constant_actor(model, 2.0)
    seen = []
    handle = model.value_dicts[0].register_forward_pre_hook(
        lambda module, inputs: seen.append(inputs[0].detach().clone()))
    loss, _, _ = model.get_loss(batch_for(model))
    loss.backward()
    handle.remove()
    # Central input = joint observations, agent ID, joint actions, Q selector.
    assert th.allclose(seen[0][:, -3:-1], th.full((4, 2), 0.96402758))
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.policy_dicts.parameters())


def test_explicit_legacy_flag_preserves_historical_raw_actor_path():
    args = arguments(legacy_raw_actor=True)
    action, _ = select_action(args, th.tensor([2.0]), exploration=False,
                              info={'log_std': th.zeros(1)})
    assert action.item() == 2.0


def reference_model(args=None):
    from models.model_registry import Model
    assert 'matd3_reference' in Model, 'reference solver must be independently selectable'
    cls = Model['matd3_reference']
    args = args or arguments()
    return cls(args, cls(args))


@pytest.mark.parametrize('shared', [False, True])
@pytest.mark.parametrize('agent_type', ['mlp', 'rnn'])
def test_reference_independent_twins_and_actor_gradient_for_each_agent(shared, agent_type):
    model = reference_model(arguments(shared_params=shared, agent_type=agent_type))
    first, second = model.value_dicts
    assert not {p.data_ptr() for p in first.parameters()} & {p.data_ptr() for p in second.parameters()}
    before = [p.clone() for p in second.parameters()]
    obs = th.zeros(2, 2, 3)
    actions = th.full((2, 2, 1), 0.4, requires_grad=True)
    q = model.value(obs, actions)
    q[0, 0].backward()
    assert actions.grad[0, 0].abs().sum() > 0
    assert actions.grad[0, 1].abs().sum() == 0
    optimizer = th.optim.SGD(first.parameters(), lr=0.01)
    optimizer.step()
    assert all(th.equal(p, saved) for p, saved in zip(second.parameters(), before))
    constant_actor(model, 2.0)
    actor_loss, critic_loss, _ = model.get_loss(batch_for(model))
    model.zero_grad()
    actor_loss.backward()
    assert th.isfinite(critic_loss)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.policy_dicts.parameters())


def test_reference_bellman_actions_come_from_target_actor_and_minimum_target_q():
    model = reference_model(arguments(target_policy_noise=0.0, gamma=0.5))
    constant_actor(model, 2.0)
    constant_actor(model.target_net, -2.0)
    seen = []
    handle = model.target_net.value_dicts[0][0].register_forward_pre_hook(
        lambda module, inputs: seen.append(inputs[0].detach().clone()))
    with th.no_grad():
        for group, output in zip(model.target_net.value_dicts, [2.0, 4.0]):
            for critic in group:
                for p in critic.parameters():
                    p.zero_()
                critic.fc3.bias.fill_(output)
        for p in model.value_dicts.parameters():
            p.zero_()
    _, loss, _ = model.get_loss(batch_for(model))
    handle.remove()
    # Each residual is (r=1 + gamma=.5 * min(2,4)) - Q=0 = 2.
    assert loss.item() == pytest.approx(8.0)
    assert th.allclose(seen[0][:, -2:], th.full((4, 2), -0.96402758))
    loss.backward()
    assert all(p.grad is None for p in model.target_net.parameters())


def test_reference_clips_smoothing_noise_then_action_and_isolates_random_streams():
    model = reference_model(arguments(target_policy_noise=100.0, target_noise_clip=0.2,
                                      target_noise_seed=13, exploration_seed=17))
    constant_actor(model.target_net, 0.0)
    obs = th.zeros(32, 2, 3)
    hidden = th.zeros(32, 2, 8)
    mask = th.ones(32, 2, 1)
    global_before = th.random.get_rng_state().clone()
    _, smoothed, _, _, _ = model.get_actions(obs, 'train', True, mask, target=True,
                                            last_hid=hidden, clip=True)
    assert smoothed.min().item() == pytest.approx(-0.2)
    assert smoothed.max().item() == pytest.approx(0.2)
    assert th.equal(global_before, th.random.get_rng_state())
    target_before = model.target_noise_generator.get_state().clone()
    model.get_actions(obs, 'train', True, mask, last_hid=hidden)
    assert th.equal(target_before, model.target_noise_generator.get_state())
    constant_actor(model.target_net, 2.0)
    _, bounded, _, _, _ = model.get_actions(obs, 'train', True, mask, target=True,
                                           last_hid=hidden, clip=True)
    assert bounded.max().item() == 1.0
    assert bounded.min().item() >= 0.7640275


def test_reference_actor_and_target_delay_count_critic_steps_not_environment_steps():
    from utilities.trainer import PGTrainer
    model = reference_model(arguments(value_update_epochs=3, policy_delay=2,
                                      policy_update_epochs=7, target_update_freq=1))
    trainer = PGTrainer(model.args, type(model), None, None)
    rows = list(zip(*batch_for(trainer.behaviour_net)))
    for row in rows:
        trainer.replay_buffer.add_experience(trainer.behaviour_net.Transition(*row))
    stat = {}
    trainer.steps = 100
    actor_before = [p.clone() for p in trainer.behaviour_net.policy_dicts.parameters()]
    target_before = [p.clone() for p in trainer.behaviour_net.target_net.parameters()]
    trainer.reference_replay_process(stat)
    assert trainer.update_counts == {'critic': 1, 'actor': 0, 'target': 0}
    assert all(th.equal(p, old) for p, old in zip(trainer.behaviour_net.policy_dicts.parameters(), actor_before))
    assert all(th.equal(p, old) for p, old in zip(trainer.behaviour_net.target_net.parameters(), target_before))
    trainer.reference_replay_process(stat)
    assert trainer.update_counts == {'critic': 2, 'actor': 1, 'target': 1}
    assert any(not th.equal(p, old) for p, old in zip(trainer.behaviour_net.policy_dicts.parameters(), actor_before))
    assert any(not th.equal(p, old) for p, old in zip(trainer.behaviour_net.target_net.parameters(), target_before))
    for target, online, old in zip(trainer.behaviour_net.target_net.policy_dicts.parameters(),
                                  trainer.behaviour_net.policy_dicts.parameters(),
                                  actor_before):
        assert th.allclose(target, old * 0.9 + online * 0.1)
    trainer.behaviour_net.transition_update(trainer, trainer.behaviour_net.Transition(*rows[0]), stat)
    assert trainer.update_counts == {'critic': 5, 'actor': 2, 'target': 2}
    assert stat['total_critic_updates'] == 5
    assert stat['total_actor_updates'] == 2
    assert stat['total_target_updates'] == 2


def test_reference_regional_action_ownership_is_bounded_and_masked():
    model = reference_model(arguments(action_dim=3))
    constant_actor(model, 2.0)
    mask = th.tensor([[[1., 0., 1.], [0., 1., 0.]]])
    action, restored, _, _, _ = model.get_actions(th.zeros(1, 2, 3), 'train', False,
                                                  mask, last_hid=th.zeros(1, 2, 8))
    assert action.shape == (1, 1, 3)
    assert th.allclose(action, th.full((1, 1, 3), 0.96402758))
    assert th.allclose(restored, mask * 0.96402758)


def test_terminal_transition_does_not_bootstrap_target_value():
    model = reference_model(arguments(target_policy_noise=0.0))
    with th.no_grad():
        for p in model.value_dicts.parameters():
            p.zero_()
        for group in model.target_net.value_dicts:
            for critic in group:
                critic.fc3.bias.fill_(100.0)
    batch = batch_for(model)._replace(done=(True, True))
    _, loss, _ = model.get_loss(batch)
    assert loss.item() == pytest.approx(2.0)


def test_reference_rollout_stores_bounded_actions_and_respects_warmup():
    from utilities.trainer import PGTrainer

    class TinyEnv:
        def reset(self):
            self.steps = 0
            self.actions = []
            return self.get_obs(), None

        def get_obs(self):
            return np.full((2, 3), self.steps * 0.1)

        def get_avail_actions(self):
            return np.ones((1, 2, 1))

        def get_num_of_agents(self):
            return 2

        def step(self, action):
            self.actions.append(action.copy())
            self.steps += 1
            return 1.0, self.steps == 4, {}

    model = reference_model(arguments(max_steps=4, action_bias=0.0, action_scale=1.0,
                                      replay_warmup=1, policy_delay=2))
    env = TinyEnv()
    trainer = PGTrainer(model.args, type(model), env, None)
    stat = {}
    trainer.behaviour_net.train_process(stat, trainer)
    assert trainer.steps == 4
    assert trainer.update_counts == {'critic': 2, 'actor': 1, 'target': 1}
    assert len(trainer.replay_buffer.buffer) == 4
    for row, action in zip(trainer.replay_buffer.buffer, env.actions):
        assert np.max(np.abs(row.action)) <= 1.0
        np.testing.assert_allclose(row.action.squeeze(), action, atol=1e-7)


def test_legacy_training_counts_real_steps_when_entropy_is_disabled():
    from utilities.trainer import PGTrainer
    trainer = PGTrainer(arguments(entr=0.0), MATD3, None, None)
    batch = batch_for(trainer.behaviour_net)
    stat = {}
    trainer.value_transition_process(stat, batch)
    trainer.policy_transition_process(stat, batch)
    assert trainer.update_counts == {'critic': 1, 'actor': 1, 'target': 0}
    assert np.isfinite(stat['mean_train_policy_loss'])


@pytest.mark.skipif(not th.cuda.is_available(), reason='CUDA is unavailable')
def test_reference_cuda_uses_device_local_noise_and_completes_optimizer_cycle():
    from utilities.trainer import PGTrainer
    model = reference_model(arguments(cuda=True))
    trainer = PGTrainer(model.args, type(model), None, None)
    for row in zip(*batch_for(trainer.behaviour_net)):
        trainer.replay_buffer.add_experience(trainer.behaviour_net.Transition(*row))
    stat = {}
    trainer.reference_replay_process(stat)
    trainer.reference_replay_process(stat)
    assert trainer.update_counts == {'critic': 2, 'actor': 1, 'target': 1}
    assert np.isfinite(stat['mean_train_policy_loss'])
    assert trainer.behaviour_net.target_noise_generator.device.type == 'cuda'


def derivative_model(**overrides):
    args = arguments(exploration_seed=31, target_noise_seed=79, **overrides)
    model = MATD3(args, MATD3(args))
    constant_actor(model, 0.0)
    constant_actor(model.target_net, 0.0)
    return model


def derivative_sample(model, *, clip=False, status='train'):
    device = model.device
    return model.get_actions(th.zeros(1, 2, 3, device=device), status, True,
                             th.ones(1, 2, 1, device=device),
                             last_hid=th.zeros(1, 2, 8, device=device), clip=clip)[0]


@pytest.mark.parametrize('cuda', [False, pytest.param(True, marks=pytest.mark.skipif(
    not th.cuda.is_available(), reason='CUDA is unavailable'))])
def test_derivative_target_sampling_does_not_advance_exploration_and_is_reproducible(cuda):
    baseline = derivative_model(cuda=cuda)
    intervened = derivative_model(cuda=cuda)
    assert th.equal(derivative_sample(baseline), derivative_sample(intervened))
    derivative_sample(intervened, clip=True)
    derivative_sample(intervened, clip=True)
    assert th.equal(derivative_sample(baseline), derivative_sample(intervened))


def test_derivative_sampling_does_not_consume_global_torch_rng():
    model = derivative_model()
    before = th.random.get_rng_state().clone()
    derivative_sample(model)
    derivative_sample(model, clip=True)
    assert th.equal(before, th.random.get_rng_state())


def test_derivative_test_action_consumes_neither_random_stream():
    baseline = derivative_model()
    intervened = derivative_model()
    derivative_sample(intervened, status='test')
    derivative_sample(intervened, status='test', clip=True)
    assert th.equal(derivative_sample(baseline), derivative_sample(intervened))
    assert th.equal(derivative_sample(baseline, clip=True), derivative_sample(intervened, clip=True))


def test_explicit_legacy_rng_mode_retains_global_normal_sample_sequence():
    model = derivative_model(legacy_raw_actor=True, solver_rng_mode='global_legacy')
    th.manual_seed(718)
    expected = th.tanh(th.distributions.Normal(th.zeros(1, 2, 1), th.ones(1, 2, 1)).rsample())
    th.manual_seed(718)
    before = th.random.get_rng_state().clone()
    actual = derivative_sample(model)
    assert th.equal(actual, expected)
    assert not th.equal(before, th.random.get_rng_state())
