import numpy as np
import torch as th
from torch import optim
from utilities.util import multinomial_entropy, get_grad_norm, normal_entropy
from utilities.replay_buffer import TransReplayBuffer, EpisodeReplayBuffer



class PGTrainer(object):
    def __init__(self, args, model, env, logger):
        self.args = args
        self.device = th.device( "cuda" if th.cuda.is_available() and self.args.cuda else "cpu" )
        self.logger = logger
        self.episodic = self.args.episodic
        if self.args.target:
            target_net = model(self.args).to(self.device)
            self.behaviour_net = model(self.args, target_net).to(self.device)
        else:
            self.behaviour_net = model(self.args).to(self.device)
        if self.args.replay:
            if not self.episodic:
                self.replay_buffer = TransReplayBuffer( int(self.args.replay_buffer_size), seed=getattr(args, 'replay_seed', None) )
            else:
                self.replay_buffer = EpisodeReplayBuffer( int(self.args.replay_buffer_size), seed=getattr(args, 'replay_seed', None) )
        self.env = env
        self.policy_optimizer = optim.RMSprop( self.behaviour_net.policy_dicts.parameters(), lr=args.policy_lrate, alpha=0.99, eps=1e-5 )
        self.value_optimizer = optim.RMSprop( self.behaviour_net.value_dicts.parameters(), lr=args.value_lrate, alpha=0.99, eps=1e-5 )
        if self.args.mixer:
            self.mixer_optimizer = optim.RMSprop( self.behaviour_net.mixer.parameters(), lr=args.mixer_lrate, alpha=0.99, eps=1e-5 )
        self.init_action = th.zeros(1, self.args.agent_num, self.args.action_dim).to(self.device)
        self.steps = 0
        self.episodes = 0
        self.entr = self.args.entr
        # A critic count is one optimizer step jointly updating the Q estimators.
        self.update_counts = {'critic': 0, 'actor': 0, 'target': 0}
        if self.args.mixer:
            self.update_counts['mixer'] = 0
        self.at_episode_boundary = True
        self.last_episode_reason = None
        self.last_episode_steps = 0

    def get_loss(self, batch):
        policy_loss, value_loss, logits = self.behaviour_net.get_loss(batch)
        return policy_loss, value_loss, logits

    def policy_compute_grad(self, stat, loss, retain_graph):
        policy_loss = loss[0]
        if self.entr > 0:
            if self.args.continuous:
                policy_loss, means, log_stds = loss
                entropy = normal_entropy( means, log_stds.exp() )
            else:
                policy_loss, logits = loss
                entropy = multinomial_entropy(logits)
            policy_loss -= self.entr * entropy
            stat['mean_train_entropy'] = entropy.item()
        policy_loss.backward(retain_graph=retain_graph)

    def value_compute_grad(self, value_loss, retain_graph):
        value_loss.backward(retain_graph=retain_graph)

    def grad_clip(self, params):
        for param in params:
            param.grad.data.clamp_(-self.args.grad_clip_eps, self.args.grad_clip_eps)

    def policy_replay_process(self, stat):
        batch = self.replay_buffer.get_batch(self.args.batch_size)
        batch = self.behaviour_net.Transition(*zip(*batch))
        self.policy_transition_process(stat, batch)

    def value_replay_process(self, stat):
        batch = self.replay_buffer.get_batch(self.args.batch_size)
        batch = self.behaviour_net.Transition(*zip(*batch))
        self.value_transition_process(stat, batch)

    def mixer_replay_process(self, stat):
        batch = self.replay_buffer.get_batch(self.args.batch_size)
        batch = self.behaviour_net.Transition(*zip(*batch))
        self.mixer_transition_process(stat, batch)

    def policy_transition_process(self, stat, trans):
        if self.args.continuous:
            policy_loss, _, logits = self.get_loss(trans)
            means, log_stds = logits
        else:
            policy_loss, _, logits = self.get_loss(trans)
        self.policy_optimizer.zero_grad()
        if self.args.continuous:
            self.policy_compute_grad(stat, (policy_loss, means, log_stds), False)
        else:
            self.policy_compute_grad(stat, (policy_loss, logits), False)
        param = self.policy_optimizer.param_groups[0]['params']
        policy_grad_norms = get_grad_norm(self.args, param)
        self.policy_optimizer.step()
        self.record_update('actor', stat)
        stat['mean_train_policy_grad_norm'] = policy_grad_norms.item() # np.array(policy_grad_norms).mean()
        stat['mean_train_policy_loss'] = policy_loss.clone().mean().item()

    def value_transition_process(self, stat, trans):
        _, value_loss, _ = self.get_loss(trans)
        self.value_optimizer.zero_grad()
        self.value_compute_grad(value_loss, False)
        param = self.value_optimizer.param_groups[0]['params']
        value_grad_norms = get_grad_norm(self.args, param)
        self.value_optimizer.step()
        self.record_update('critic', stat)
        stat['mean_train_value_grad_norm'] = value_grad_norms.item()
        stat['mean_train_value_loss'] = value_loss.clone().mean().item()

    def mixer_transition_process(self, stat, trans):
        _, value_loss, _ = self.get_loss(trans)
        self.mixer_optimizer.zero_grad()
        self.value_compute_grad(value_loss, False)
        param = self.mixer_optimizer.param_groups[0]['params']
        mixer_grad_norms = get_grad_norm(self.args, param)
        self.mixer_optimizer.step()
        self.record_update('mixer', stat)
        stat['mean_train_mixer_grad_norm'] = mixer_grad_norms.item()
        stat['mean_train_mixer_loss'] = value_loss.clone().mean().item()

    def record_update(self, kind, stat):
        self.update_counts[kind] += 1
        for name, count in self.update_counts.items():
            stat['total_' + name + '_updates'] = count

    def reference_replay_process(self, stat):
        """One real critic step, optionally followed by actor + both targets.

        Reuse this sampled batch for the delayed actor, unpacking/normalizing
        once. The counter is advanced after optimizer.step(), never on sampling.
        """
        net = self.behaviour_net
        batch = net.Transition(*zip(*self.replay_buffer.get_batch(self.args.batch_size)))
        data = net.unpack_data(batch)
        self.value_optimizer.zero_grad(set_to_none=True)
        value_loss = net.critic_loss(data)
        value_loss.backward()
        stat['mean_train_value_grad_norm'] = get_grad_norm(
            self.args, self.value_optimizer.param_groups[0]['params']).item()
        self.value_optimizer.step()
        self.record_update('critic', stat)
        stat['mean_train_value_loss'] = value_loss.item()
        if self.update_counts['critic'] % self.args.policy_delay == 0:
            # Keep dQ/da while avoiding unused critic parameter gradients.
            net.value_dicts.requires_grad_(False)
            try:
                self.policy_optimizer.zero_grad(set_to_none=True)
                policy_loss, _ = net.actor_loss(data)
                policy_loss.backward()
                stat['mean_train_policy_grad_norm'] = get_grad_norm(
                    self.args, self.policy_optimizer.param_groups[0]['params']).item()
                self.policy_optimizer.step()
                self.record_update('actor', stat)
                stat['mean_train_policy_loss'] = policy_loss.item()
            finally:
                net.value_dicts.requires_grad_(True)
            net.update_target()
            self.record_update('target', stat)

    def run(self, stat, episode):
        self.behaviour_net.train_process(stat, self)
        if (episode%self.args.eval_freq == self.args.eval_freq-1) or (episode == 0):
            self.behaviour_net.evaluation(stat, self)

    def logging(self, stat):
        for k, v in stat.items():
            # Per-inverter evidence is retained in JSON, not coerced to a scalar.
            if np.asarray(v).ndim == 0:
                self.logger.add_scalar('data/' + k, v, self.episodes)

    def print_info(self, stat):
        string = [f'\nEpisode: {self.episodes}']
        for k, v in stat.items():
            string.append( k + f': {v:2.4f}' )
        string = "\n".join(string)
        print (string)
