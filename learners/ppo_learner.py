from utils.rl_utils import build_gae_targets, categorical_entropy, ValueNorm, build_vtrace_gae_targets
import torch
from torch.optim import Adam
from modules.critics import REGISTRY as critic_resigtry


class PPOLearner:
    def __init__(self, args, mac, queue_center=None):
        self.args = args
        self.mac = mac
        self.queue_center = queue_center

        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.device = args.train_device
        self.train_batch_size = args.train_batch_size
        self.last_target_update_step = 0
        self.critic_training_steps = 0
        self.critic = critic_resigtry[args.critic_type](args)
        self.cuda()
        self.params = list(mac.parameters()) + list(self.critic.parameters())

        self.optimiser = Adam(params=self.params, lr=args.lr)
        self.last_lr = args.lr

        self.use_value_norm = getattr(self.args, "use_value_norm", False)
        if self.use_value_norm:
            self.value_norm = ValueNorm(1, device=self.device)

    def train(self, batch, episode_num):
        max_seq_length = batch["length"]
        if self.args.use_individual_rewards:
            rewards = batch["individual_rewards"][:, :-1]
        else:
            rewards = batch["reward"][:, :-1].unsqueeze(2).repeat(1, 1, self.n_agents, 1)
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"][:, :-1]

        old_probs = batch["probs"][:, :-1]
        old_probs[avail_actions == 0] = 1e-10
        old_logprob = torch.log(torch.gather(old_probs, dim=3, index=actions)).detach()

        mask_agent = mask.unsqueeze(2).repeat(1, 1, self.n_agents, 1)
        with torch.no_grad():
            if "rnn" in self.args.critic_type:
                old_values = []
                hidden_states = torch.zeros(self.train_batch_size, self.args.n_agents, self.args.hidden_dim).to(actions.device)
                for t in range(max_seq_length):
                    inputs = {}
                    inputs["obs"] = batch["obs"][:, t]
                    inputs["avail_actions"] = batch["avail_actions"][:, t]
                    if self.args.obs_last_action:
                        if t == 0:
                            inputs["actions_onehot"] = torch.zeros_like(inputs["avail_actions"])
                        else:
                            inputs["actions_onehot"] = batch["actions_onehot"][:, t - 1]
                    outputs, hidden_states = self.critic(inputs, hidden_states)
                    old_values.append(outputs)
                old_values = torch.stack(old_values, dim=1)
            elif "mlp" in self.args.critic_type:
                old_values = self.critic(batch)

            if self.use_value_norm:
                value_shape = old_values.shape
                norm_old_values = self.value_norm.denormalize(old_values.view(-1)).view(value_shape)

            if not self.args.use_vtrace:

                advantages, targets = build_gae_targets(rewards, mask_agent, norm_old_values, self.args.gamma, self.args.gae_lambda)

                if self.use_value_norm:
                    targets_shape = targets.shape
                    targets = targets.reshape(-1)
                    self.value_norm.update(targets)
                    targets = self.value_norm.normalize(targets).view(targets_shape)

                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

        # PPO Loss
        for _ in range(self.args.mini_epochs):
            # Actor
            pi = []
            hidden_states = self.mac.init_hidden(self.train_batch_size)
            for t in range(max_seq_length - 1):
                inputs = {}
                inputs["hidden_states"] = hidden_states
                inputs["obs"] = batch["obs"][:, t]
                inputs["avail_actions"] = batch["avail_actions"][:, t]
                if self.args.obs_last_action:
                    if t == 0:
                        inputs["actions_onehot"] = torch.zeros_like(inputs["avail_actions"])
                    else:
                        inputs["actions_onehot"] = batch["actions_onehot"][:, t - 1]
                agent_outs, hidden_states = self.mac.inference(inputs)
                pi.append(agent_outs)
            pi = torch.stack(pi, dim=1)
            pi[avail_actions == 0] = 1e-10
            pi_taken = torch.gather(pi, dim=3, index=actions)
            log_pi_taken = torch.log(pi_taken)
            ratios = torch.exp(log_pi_taken - old_logprob)

            if self.args.use_vtrace:
                advantages, targets = build_vtrace_gae_targets(ratios, rewards, mask_agent, norm_old_values, self.args.gamma, self.args.gae_lambda)
                if self.use_value_norm:
                    targets_shape = targets.shape
                    targets = targets.reshape(-1)
                    self.value_norm.update(targets)
                    targets = self.value_norm.normalize(targets).view(targets_shape)

                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

            # Critic
            if "rnn" in self.args.critic_type:
                values = []
                hidden_states = torch.zeros(self.train_batch_size, self.args.n_agents, self.args.hidden_dim).to(actions.device)
                for t in range(max_seq_length - 1):
                    inputs = {}
                    inputs["obs"] = batch["obs"][:, t]
                    inputs["avail_actions"] = batch["avail_actions"][:, t]
                    if self.args.obs_last_action:
                        if t == 0:
                            inputs["actions_onehot"] = torch.zeros_like(inputs["avail_actions"])
                        else:
                            inputs["actions_onehot"] = batch["actions_onehot"][:, t - 1]
                    outputs, hidden_states = self.critic(inputs, hidden_states)
                    values.append(outputs)
                values = torch.stack(values, dim=1)
            elif "mlp" in self.args.critic_type:
                values = self.critic(batch)[:, :-1]

            # value clip
            values_clipped = old_values[:, :-1] + (values - old_values[:, :-1]).clamp(-self.args.eps_clip, self.args.eps_clip)

            # 0-out the targets that came from padded data
            td_error = torch.max((values - targets.detach()) ** 2, (values_clipped - targets.detach()) ** 2)
            masked_td_error = td_error * mask_agent
            critic_loss = 0.5 * masked_td_error.sum() / mask_agent.sum()

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.args.eps_clip, 1 + self.args.eps_clip) * advantages
            if self.args.dual_clip:
                clip1 = torch.min(surr1, surr2)
                clip2 = torch.max(clip1, self.args.dual_clip * advantages)
                actor_loss = -(torch.where(advantages < 0, clip2, clip1) * mask_agent).sum() / mask_agent.sum()
            else:
                actor_loss = -(torch.min(surr1, surr2) * mask_agent).sum() / mask_agent.sum()

            # entropy
            entropy_loss = categorical_entropy(pi).mean(-1, keepdim=True)  # mean over agents
            entropy_loss[mask == 0] = 0  # fill nan
            entropy_loss = (entropy_loss * mask).sum() / mask.sum()
            loss = actor_loss + self.args.critic_coef * critic_loss - self.args.entropy * entropy_loss / entropy_loss.item()

            # Optimise agents
            self.optimiser.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
            self.optimiser.step()
            self.mac.agent_version += 1

    def cuda(self):
        if self.args.use_dp:
            self.critic = torch.nn.DataParallel(self.critic)
            self.mac.set_dp()

        self.mac.to(self.device)
        self.critic.to(self.device)

    def save_models(self, path):
        self.mac.save_models(path)
        torch.save(self.optimiser.state_dict(), "{}/agent_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.optimiser.load_state_dict(torch.load("{}/agent_opt.th".format(path), map_location=lambda storage, loc: storage))
