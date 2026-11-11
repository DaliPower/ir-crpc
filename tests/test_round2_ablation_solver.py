"""Mechanism and baseline invariance tests; synthetic transitions only."""
from pathlib import Path
from types import SimpleNamespace
import runpy

import pytest
import torch

from models.matd3 import MATD3
from models.matd3_reference import MATD3Reference
from models.matd3_round2_ablation import MATD3Round2Ablation
from utilities.trainer import PGTrainer
from utilities.util import select_action

HELPERS = runpy.run_path(str(Path(__file__).with_name('test_revision_solver.py')))
arguments, batch_for, constant_actor = (HELPERS[k] for k in ('arguments', 'batch_for', 'constant_actor'))


def make(variant='A0', cls=MATD3Round2Ablation, **kwargs):
    delay = 1 if variant in ('A1', 'A3') else 2
    args = arguments(round2_variant=variant, policy_delay=delay, **kwargs)
    return cls(args, cls(args))


def test_a0_preserves_reference_parameters_losses_actions_and_rng_exactly():
    torch.manual_seed(37)
    reference = make(cls=MATD3Reference)
    torch.manual_seed(37)
    actual = make()
    assert reference.state_dict().keys() == actual.state_dict().keys()
    for key, tensor in reference.state_dict().items():
        assert torch.equal(tensor, actual.state_dict()[key]), key
    data = batch_for(reference)
    rl = reference.get_loss(data)
    al = actual.get_loss(data)
    assert torch.equal(rl[0], al[0]) and torch.equal(rl[1], al[1])
    for gen in ('target_noise_generator', 'exploration_generator'):
        assert torch.equal(getattr(reference, gen).get_state(), getattr(actual, gen).get_state())
    al[0].backward(); rl[0].backward()
    for (name, p), (other, q) in zip(reference.named_parameters(), actual.named_parameters()):
        assert name == other
        if p.grad is not None:
            assert torch.equal(p.grad, q.grad), name


@pytest.mark.parametrize('variant,expected', [('A0',True), ('A1',True), ('A2',False), ('A3',False), ('A4',True), ('A5',True)])
def test_td_actor_source_is_the_only_source_switch(variant, expected):
    net = make(variant)
    calls = []
    real = net.get_actions
    def capture(*a, **kw):
        calls.append(kw['target'])
        return real(*a, **kw)
    net.get_actions = capture
    net.critic_loss(net.unpack_data(batch_for(net)))
    assert calls == [expected]


@pytest.mark.parametrize('variant,actor_count', [('A0',2),('A1',4),('A2',2),('A3',4),('A4',2),('A5',2)])
def test_joint_actor_target_schedule_uses_completed_critic_steps(variant, actor_count):
    net = make(variant)
    trainer = PGTrainer(net.args, MATD3Round2Ablation, None, None)
    for row in zip(*batch_for(net)):
        trainer.replay_buffer.add_experience(net.Transition(*row))
    for _ in range(4):
        trainer.reference_replay_process({})
    assert trainer.update_counts == {'critic':4, 'actor':actor_count, 'target':actor_count}


def test_a4_exact_legacy_pre_tanh_draw_only_on_td_path():
    net = make('A4')
    constant_actor(net.target_net, 2.0)
    data = net.unpack_data(batch_for(net))
    obs, mask, hidden = data[6], data[9], data[11]
    means, log_std, _ = net.target_net.policy(obs, last_hid=hidden)
    generator = torch.Generator().set_state(net.target_noise_generator.get_state())
    expected, _ = select_action(net.args, means, exploration=True,
        info={'clip':True,'log_std':log_std}, generator=generator)
    exploration_state = net.exploration_generator.get_state().clone()
    result = net.get_actions(obs, 'train', True, mask, target=True, last_hid=hidden, clip=True)
    assert torch.equal(result[1], expected)
    assert torch.equal(exploration_state, net.exploration_generator.get_state())
    assert result[1].abs().max() <= 1
    # Behavior exploration remains post-tanh reference noise, not historical noise.
    torch.manual_seed(111); base = make('A0')
    torch.manual_seed(111); alt = make('A4')
    a = base.get_actions(obs, 'train', True, mask, last_hid=hidden)[1]
    b = alt.get_actions(obs, 'train', True, mask, last_hid=hidden)[1]
    assert torch.equal(a, b)


@pytest.mark.parametrize('shared', [True,False])
def test_a5_reuses_exact_historical_conditional_q_without_loss_scaling_change(shared):
    net = make('A5', shared_params=shared)
    legacy = MATD3(net.args)
    legacy.value_dicts.load_state_dict(net.value_dicts.state_dict())
    obs, act = torch.randn(2,2,3), torch.randn(2,2,1)
    assert torch.equal(net.value(obs,act), legacy.value(obs,act))
    assert len(net.value_dicts) == (1 if shared else 2)
    data = net.unpack_data(batch_for(net))
    state = net.target_noise_generator.get_state()
    actual = net.critic_loss(data)
    net.target_noise_generator.set_state(state)
    with torch.no_grad():
        nxt = net.get_actions(data[6], 'train', True, data[9], target=True, last_hid=data[11], clip=True)[1]
        q1t,q2t = net.target_net.value(data[6],nxt).chunk(2)
        target = data[5] + net.args.gamma * (1-data[7]) * torch.minimum(q1t.squeeze(-1),q2t.squeeze(-1))
    q1,q2 = net.value(data[0],data[1]).chunk(2)
    expected = (q1.squeeze(-1)-target).square().mean() + (q2.squeeze(-1)-target).square().mean()
    assert torch.equal(actual,expected)


def test_invalid_variant_and_inconsistent_delay_are_rejected():
    with pytest.raises(ValueError): make('A6')
    with pytest.raises(ValueError):
        MATD3Round2Ablation(arguments(round2_variant='A1', policy_delay=2))
