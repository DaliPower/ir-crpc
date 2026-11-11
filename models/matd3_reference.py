"""MATD3 reference mechanisms adapted to MAPDN's local recurrent policies.

Sources: Ackermann et al., https://arxiv.org/abs/1910.01465v2, sec. 4;
Fujimoto et al., https://github.com/sfujim/TD3/blob/master/TD3.py.
This is an adaptation, not a reproduction of either paper's experiments.

Each Q sees joint observations/actions. Q1 and Q2 never share parameters;
shared_params optionally shares each Q across agent IDs and shares the actor
across agents. Only agent i's action receives the gradient from Q_i; the shared
actor accumulates these local paths. Regional action masks assign each physical
device to one agent. Actors use local observations (and optionally agent IDs).

RNN updates use detached behavior hidden states stored in transition replay,
including for the target actor. There is no sequence burn-in or BPTT: target
hidden histories are an off-policy approximation, not recomputed target history.
Critics remain feed-forward. These choices must be disclosed in comparisons.
The common MAPDN trainer retains RMSprop for optimizer comparability; it is not
the Adam optimizer used in the cited benchmark implementations.
"""
import torch as th
from torch import nn
from critics.mlp_critic import MLPCritic
from models.model import Model


class MATD3Reference(Model):
    critic_step_schedule = True

    def __init__(self, args, target_net=None):
        super().__init__(args)
        if not args.continuous or not args.target or args.gaussian_policy:
            raise ValueError('matd3_reference requires continuous deterministic actors and targets')
        if not args.replay or args.episodic or args.mixer:
            raise ValueError('matd3_reference requires transition replay without a mixer')
        if not args.action_enforcebound:
            raise ValueError('matd3_reference requires normalized bounded actions')
        if args.entr != 0 or args.normalize_advantages:
            raise ValueError('matd3_reference uses an unregularized deterministic actor objective')
        if args.policy_delay < 1 or int(args.policy_delay) != args.policy_delay:
            raise ValueError('policy_delay must be a positive integer number of critic steps')
        for key in ('target_policy_noise', 'target_noise_clip', 'exploration_noise'):
            if getattr(args, key) < 0:
                raise ValueError(key + ' must be nonnegative')
        self.construct_model()
        self.apply(self.init_weights)
        if target_net is not None:
            self.target_net = target_net
            self.reload_params_to_target()
            self.target_net.requires_grad_(False)
        self.to(self.device)
        # Independent streams: evaluation and exploration cannot advance target noise.
        self.exploration_generator = th.Generator(device=self.device)
        self.target_noise_generator = th.Generator(device=self.device)
        seed = int(getattr(args, 'seed', 0))
        self.exploration_generator.manual_seed(int(getattr(args, 'exploration_seed', seed + 1)))
        self.target_noise_generator.manual_seed(int(getattr(args, 'target_noise_seed', seed + 2)))

    def construct_model(self):
        self.construct_policy_net()
        self.construct_value_net()

    def construct_value_net(self):
        size = (self.obs_dim + self.act_dim) * self.n_
        if self.args.agent_id:
            size += self.n_
        count = 1 if self.args.shared_params else self.n_
        self.value_dicts = nn.ModuleList([
            nn.ModuleList([MLPCritic(size, 1, self.args) for _ in range(count)])
            for _ in range(2)
        ])

    def value(self, obs, act):
        batch = obs.shape[0]
        ids = th.eye(self.n_, device=obs.device).unsqueeze(0).expand(batch, -1, -1)
        joint_obs = obs.reshape(batch, 1, -1).expand(-1, self.n_, -1)
        if self.args.agent_id:
            joint_obs = th.cat((joint_obs, ids), dim=-1)
        joint_action = act.unsqueeze(1).expand(-1, self.n_, -1, -1)
        own = ids.unsqueeze(-1)
        joint_action = own * joint_action + (1 - own) * joint_action.detach()
        inputs = th.cat((joint_obs, joint_action.reshape(batch, self.n_, -1)), dim=-1)
        outputs = []
        for group in self.value_dicts:
            if self.args.shared_params:
                q, _ = group[0](inputs.reshape(batch * self.n_, -1), None)
                q = q.reshape(batch, self.n_, 1)
            else:
                q = th.stack([critic(inputs[:, i], None)[0] for i, critic in enumerate(group)], dim=1)
            outputs.append(q)
        # Preserve the base model's rollout/Transition value layout.
        return th.cat(outputs, dim=0)

    def get_actions(self, obs, status, exploration, actions_avail, target=False,
                    last_hid=None, clip=False):
        policy_model = self.target_net if target else self
        if last_hid is None:
            last_hid = obs.new_zeros(obs.shape[0], self.n_, self.hid_dim)
        means, log_stds, hidden = policy_model.policy(obs, last_hid=last_hid.detach())
        mask = th.as_tensor(actions_avail, device=obs.device, dtype=means.dtype).reshape(
            -1, self.n_, self.act_dim)
        if self.act_dim > 1:
            if th.any(mask.sum(dim=1) > 1):
                raise ValueError('regional action masks must assign each device to at most one agent')
            # MAPDN regional actors have a padded vector of globally indexed devices.
            logits = (means * mask).sum(dim=1, keepdim=True)
            physical_mask = mask.sum(dim=1, keepdim=True)
        else:
            logits = means
            physical_mask = mask
        actions = th.tanh(logits)
        if status == 'train' and exploration:
            generator = self.target_noise_generator if clip else self.exploration_generator
            sigma = self.args.target_policy_noise if clip else self.args.exploration_noise
            noise = th.randn(actions.shape, generator=generator, device=actions.device,
                             dtype=actions.dtype) * sigma
            if clip:
                noise = noise.clamp(-self.args.target_noise_clip, self.args.target_noise_clip)
            actions = (actions + noise).clamp(-1, 1)
        actions = actions * physical_mask
        restored = actions * mask
        # DPG does not use log probabilities; base rollout requires a tensor to store.
        return actions, restored, th.zeros_like(restored), (means, log_stds), hidden

    def critic_loss(self, data):
        state, actions, _, _, _, rewards, next_state, done, _, mask, _, hids = data
        with th.no_grad():
            _, next_actions, _, _, _ = self.get_actions(
                next_state, 'train', True, mask, target=True, last_hid=hids, clip=True)
            q1_target, q2_target = self.target_net.value(next_state, next_actions).chunk(2, dim=0)
            target = rewards + self.args.gamma * (1 - done) * th.minimum(
                q1_target.squeeze(-1), q2_target.squeeze(-1))
        q1, q2 = self.value(state, actions).chunk(2, dim=0)
        return (q1.squeeze(-1) - target).square().mean() + (q2.squeeze(-1) - target).square().mean()

    def actor_loss(self, data):
        state, _, _, _, _, _, _, _, _, mask, last_hids, _ = data
        _, actions, _, outputs, _ = self.get_actions(
            state, 'train', False, mask, last_hid=last_hids)
        q1, _ = self.value(state, actions).chunk(2, dim=0)
        return -q1.mean(), outputs

    def get_loss(self, batch):
        data = self.unpack_data(batch)
        critic_loss = self.critic_loss(data)
        actor_loss, outputs = self.actor_loss(data)
        return actor_loss, critic_loss, outputs

    def transition_update(self, trainer, trans, stat):
        trainer.replay_buffer.add_experience(trans)
        if (trainer.steps > self.args.replay_warmup
                and len(trainer.replay_buffer.buffer) >= self.args.batch_size
                and trainer.steps % self.args.behaviour_update_freq == 0):
            for _ in range(self.args.value_update_epochs):
                trainer.reference_replay_process(stat)
        # No environment-step target updates and no extra policy_update_epochs loop.
