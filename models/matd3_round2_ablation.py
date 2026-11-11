"""Controlled mechanisms on the submitted adapted reference, not canonical MATD3.

A0 is a dispatch-only subclass with identical parameters, RNG draws, loss and
optimizer path. A1/A3 change the existing joint actor/target delay parameter.
A2/A3 change the TD actor source. A4 substitutes the historical pre-tanh Normal
draw (the bounded historical branch ignores clip_c). A5 reuses the historical
conditional shared-Q construction and forward, retaining reference loss scaling.
Behavior exploration, bounded actor objective, RNN history and RMSprop are fixed.
"""
import torch

from models.matd3 import MATD3
from models.matd3_reference import MATD3Reference
from utilities.util import select_action


VARIANTS = {
    'A0': {'policy_delay':2, 'td_actor':'target', 'td_noise':'reference', 'critic':'independent'},
    'A1': {'policy_delay':1, 'td_actor':'target', 'td_noise':'reference', 'critic':'independent'},
    'A2': {'policy_delay':2, 'td_actor':'online', 'td_noise':'reference', 'critic':'independent'},
    'A3': {'policy_delay':1, 'td_actor':'online', 'td_noise':'reference', 'critic':'independent'},
    'A4': {'policy_delay':2, 'td_actor':'target', 'td_noise':'historical_pre_tanh', 'critic':'independent'},
    'A5': {'policy_delay':2, 'td_actor':'target', 'td_noise':'reference', 'critic':'conditional_shared'},
}


class MATD3Round2Ablation(MATD3Reference):
    def __init__(self, args, target_net=None):
        variant = getattr(args, 'round2_variant', None)
        if variant not in VARIANTS:
            raise ValueError('round2_variant must be A0..A5')
        self.mechanisms = VARIANTS[variant]
        if args.policy_delay != self.mechanisms['policy_delay']:
            raise ValueError('policy_delay differs from registered variant')
        super().__init__(args, target_net)

    def construct_value_net(self):
        if self.mechanisms['critic'] == 'conditional_shared':
            return MATD3.construct_value_net(self)
        return super().construct_value_net()

    def value(self, obs, act):
        if self.mechanisms['critic'] == 'conditional_shared':
            return MATD3.value(self, obs, act)
        return super().value(obs, act)

    def critic_loss(self, data):
        if self.mechanisms['td_actor'] == 'target':
            return super().critic_loss(data)
        state, actions, _, _, _, rewards, next_state, done, _, mask, _, hids = data
        with torch.no_grad():
            _, next_actions, _, _, _ = self.get_actions(
                next_state, 'train', True, mask, target=False, last_hid=hids, clip=True)
            q1t, q2t = self.target_net.value(next_state, next_actions).chunk(2, dim=0)
            target = rewards + self.args.gamma * (1-done) * torch.minimum(q1t.squeeze(-1),q2t.squeeze(-1))
        q1,q2 = self.value(state,actions).chunk(2,dim=0)
        return (q1.squeeze(-1)-target).square().mean() + (q2.squeeze(-1)-target).square().mean()

    def get_actions(self, obs, status, exploration, actions_avail, target=False,
                    last_hid=None, clip=False):
        if not (self.mechanisms['td_noise'] == 'historical_pre_tanh'
                and status == 'train' and exploration and clip):
            return super().get_actions(obs,status,exploration,actions_avail,
                target=target,last_hid=last_hid,clip=clip)
        policy_model = self.target_net if target else self
        if last_hid is None:
            last_hid = obs.new_zeros(obs.shape[0],self.n_,self.hid_dim)
        means, log_stds, hidden = policy_model.policy(obs,last_hid=last_hid.detach())
        mask = torch.as_tensor(actions_avail,device=obs.device,dtype=means.dtype).reshape(-1,self.n_,self.act_dim)
        if self.act_dim > 1 and torch.any(mask.sum(dim=1)>1):
            raise ValueError('regional action masks must assign each device to at most one agent')
        means = means * mask
        log_stds = log_stds * mask
        reduced_means = means.sum(dim=1,keepdim=True) if self.act_dim>1 else means
        reduced_logs = log_stds.sum(dim=1,keepdim=True) if self.act_dim>1 else log_stds
        actions,_ = select_action(self.args,reduced_means,status='train',exploration=True,
            info={'clip':True,'log_std':reduced_logs},generator=self.target_noise_generator)
        physical_mask = mask.sum(dim=1,keepdim=True) if self.act_dim>1 else mask
        actions = actions * physical_mask
        restored = actions * mask
        return actions,restored,torch.zeros_like(restored),(means,log_stds),hidden
