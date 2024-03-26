import torch
from copy import deepcopy
from torch.optim import Adam
from policy.base.base_policy import BasePolicy
from .q_net import mixer_REGISTRY
from utils.utils import build_q_lambda_targets, build_td_lambda_targets

class Qpolicy(BasePolicy):
    def __init__(self, args, mac):
        super().__init__(args, mac)
        self.lr = args.lr
        self.mixer = mixer_REGISTRY[args.mixer](args)
        self.params = mac.agent_parameters + list(self.mixer.parameters())
        self.optimizer = Adam(params=self.params, lr=args.lr)
        self.target_mac = deepcopy(mac)
        self.target_mixer = deepcopy(self.mixer)
        
        self.mac.to(self.device)
        self.target_mac.to(self.device)
        self.mixer.to(self.device)
        self.target_mixer.to(self.device)

        self.last_target_update_episode = 0
        
    def train(self, batch, episodes, steps):
        # self.lr = self.lr_decay(self.lr, steps)
        b, length, a, _ = batch["obs"].shape
        obs = batch["obs"]
        states = batch["states"]
        rewards = batch["rewards"][:, :-1]
        actions = batch["actions"]
        available_actions = batch["available_actions"]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        mac_out = []
        hidden_states = self.mac.init_hidden_states(self.args.train_batch_size)
        for t in range(length):
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
            agent_outs, hidden_states = self.mac.get_agent_outputs(inputs)
            mac_out.append(agent_outs)
        mac_out = torch.stack(mac_out, dim=1)
        chosen_action_qvals = torch.gather(mac_out[:, :-1], dim=3, index=actions[:, :-1].unsqueeze(-1)).squeeze(3)
        with torch.no_grad():
            target_mac_out = []
            hidden_states = self.target_mac.init_hidden_states(self.args.train_batch_size)
            for t in range(length):
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
                agent_outs, hidden_states = self.target_mac.get_agent_outputs(inputs)
                target_mac_out.append(agent_outs)
            target_mac_out = torch.stack(target_mac_out, dim=1)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[available_actions == 0] = -9999999
            cur_max_actions = mac_out_detach.max(dim=3, keepdim=True)[1]
            target_max_qvals = torch.gather(target_mac_out, 3, cur_max_actions).squeeze(3)

            target_max_qvals = self.target_mixer(target_max_qvals, states)

            if getattr(self.args, "q_lambda", False):
                qvals = torch.gather(target_mac_out, 3, batch["actions"]).squeeze(3)
                qvals = self.target_mixer(qvals, states)
                targets = build_q_lambda_targets(rewards, terminated, mask, target_max_qvals, qvals, self.args.gamma, self.args.td_lambda)
            else:
                targets = build_td_lambda_targets(rewards, terminated, mask, target_max_qvals, self.args.n_agents, self.args.gamma, self.args.td_lambda)

        chosen_action_qvals = self.mixer(chosen_action_qvals, states[:, :-1])

        td_error = chosen_action_qvals - targets.detach()
        td_error2 = 0.5 * td_error.pow(2)

        mask = mask.expand_as(td_error2)
        masked_td_error = td_error2 * mask

        loss = masked_td_error.sum() / mask.sum()

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.params, self.args.grad_norm_clip
        )
        self.optimizer.step()
        self.mac.agent_version += 1

        if (episodes - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episodes

        mask_elems = mask.sum().item()
        metrics = {}
        metrics["loss_td"] = loss.item()
        metrics["grad_norm"] = grad_norm
        metrics["td_error_abs"] = (masked_td_error.abs().sum().item() / mask_elems)
        metrics["q_taken_mean"] = (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents)
        metrics["target_mean"] = (targets * mask).sum().item() / (mask_elems * self.args.n_agents)
        return metrics

    def _update_targets(self):
        self.target_mac.load_agent_state_dict(self.mac.agent_state_dict)
        self.target_mixer.load_state_dict(self.mixer.state_dict())