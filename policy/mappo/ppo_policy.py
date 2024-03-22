import torch
from torch.optim import Adam
from policy.base.base_policy import BasePolicy
from .ppo_net import critic_REGISTRY
from utils.utils import build_gae_targets, build_vtrace_gae_targets

class PPOPolicy(BasePolicy):
    def __init__(self, args, mac):
        super().__init__(args, mac)
        self.lr = args.lr
        self.critic = critic_REGISTRY[args.critic_type](args)
        self.params = mac.agent_parameters + list(self.critic.parameters())
        self.optimizer = Adam(params=self.params, lr=args.lr)
        self.mac.to(self.device)
        self.critic.to(self.device)


    def train(self, batch, episodes, steps):
        self.lr = self.lr_decay(self.lr, steps)
        b, length, a, _ = batch["obs"].shape

        obs = batch["obs"]
        states = batch["states"]
        if self.args.use_individual_rewards:
            rewards = batch["individual_rewards"][:, :-1]
        else:
            rewards = batch["rewards"][:, :-1].unsqueeze(2).repeat(1, 1, self.args.n_agents, 1)
        actions = batch["actions"][:, :-1]
        available_actions = batch["available_actions"][:, :-1]
        old_log_probs = batch["action_log_probs"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        mask_agent = mask.unsqueeze(2).repeat(1, 1, self.args.n_agents, 1)

        with torch.no_grad():
            old_values = []
            # just uesd for rnn type
            if "independent_obs" in self.args.critic_inputs:
                hidden_states = torch.zeros(b, a, self.args.hidden_dim).to(self.device)
            elif "central_obs" in self.args.critic_inputs:
                hidden_states = torch.zeros(b, self.args.hidden_dim).to(self.device)
            for t in range(length):
                inputs = {
                    "obs": obs[:, t],
                    "hidden_states": hidden_states
                    }
                if "state" in self.args.critic_inputs:
                    inputs["state"] = states[:, t]
                if "last_action" in self.args.critic_inputs:
                    if t == 0:
                        inputs["last_actions"] = torch.zeros(b, a, self.args.n_actions).to(self.device)
                    else:
                        inputs["last_actions"] = torch.nn.functional.one_hot(actions[:, t-1], self.args.n_actions).to(self.device)
                value, hidden_states = self.critic(inputs)
                old_values.append(value)
            old_values = torch.stack(old_values, dim=1)

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

        for _ in range(self.args.mini_epochs):
            log_pi_list = []
            entropy_list = []
            hidden_states = torch.zeros(b, a, self.args.hidden_dim).to(self.device)
            for t in range(length - 1):
                inputs = {
                    "obs": obs[:, t],
                    "hidden_states": hidden_states,
                    "available_actions": available_actions[:, t]
                }
                if "last_action" in self.args.agent_inputs:
                    if t == 0:
                        inputs["last_actions"] = torch.zeros(b, a, self.args.n_actions).to(self.device)
                    else:
                        inputs["last_actions"] = torch.nn.functional.one_hot(actions[:, t-1], self.args.n_actions).to(self.device)
                log_probs, entropy, hidden_states = self.mac.make_distribution(inputs, actions[:,t])
                log_pi_list.append(log_probs)
                entropy_list.append(entropy)
            log_probs = torch.stack(log_pi_list, dim=1)
            entropy = torch.stack(entropy_list, dim=1)
            if self.args.env_type == "continuous":
                ratios = torch.exp(log_probs.sum(-1) - old_log_probs.sum(-1)).unsqueeze(-1)
                entropy = entropy.sum(-1)
            else:
                ratios = torch.exp(log_probs - old_log_probs).unsqueeze(-1)

            if self.args.use_vtrace:
                advantages, targets = build_vtrace_gae_targets(ratios, rewards, mask_agent, norm_old_values, self.args.gamma, self.args.gae_lambda)
                if self.use_value_norm:
                    targets_shape = targets.shape
                    targets = targets.reshape(-1)
                    self.value_norm.update(targets)
                    targets = self.value_norm.normalize(targets).view(targets_shape)

                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
            
            values = []
            # just uesd for rnn type
            if "independent_obs" in self.args.critic_inputs:
                hidden_states = torch.zeros(b, a, self.args.hidden_dim).to(self.device)
            elif "central_obs" in self.args.critic_inputs:
                hidden_states = torch.zeros(b, self.args.hidden_dim).to(self.device)
            for t in range(length - 1):
                inputs = {
                    "obs": obs[:, t],
                    "hidden_states": hidden_states
                    }
                if "state" in self.args.critic_inputs:
                    inputs["states"] = states[:, t]
                if "last_action" in self.args.critic_inputs:
                    if t == 0:
                        inputs["last_actions"] = torch.zeros(b, a, self.args.n_actions).to(self.device)
                    else:
                        inputs["last_actions"] = torch.nn.functional.one_hot(actions[:, t-1], self.args.n_actions).to(self.device)
                value, hidden_states = self.critic(inputs)
                values.append(value)
            values = torch.stack(values, dim=1)
            values_clipped = old_values[:, :-1] + (values - old_values[:, :-1]).clamp(-self.args.eps_clip, self.args.eps_clip)
            # 0-out the targets that came from padded data
            td_error = torch.max((values - targets.detach()) ** 2, (values_clipped - targets.detach()) ** 2, )
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
            entropy_loss = entropy.mean(-1, keepdim=True)  # mean over agents
            entropy_loss[mask == 0] = 0 
            entropy_loss = (entropy_loss * mask).sum() / mask.sum()
            loss = actor_loss + self.args.critic_coef * critic_loss - self.args.entropy * entropy_loss / entropy_loss.item()

            # Optimise agents
            self.optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
            self.optimizer.step()
            self.mac.agent_version += 1
                
        metrics = {}
        mask_elems = mask_agent.sum().item()
        metrics["advantage_mean"] = (advantages * mask_agent).sum().item() / mask_elems
        metrics["actor_loss"] = actor_loss.item()
        metrics["entropy_loss"] = entropy_loss.item()
        metrics["grad_norm"] = grad_norm
        metrics["critic_loss"] = critic_loss.item()
        metrics["target_mean"] = (targets * mask_agent).sum().item() / mask_elems

        return metrics