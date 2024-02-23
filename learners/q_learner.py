import copy
from modules.mixers.qmix import QMixer
from modules.mixers.vdn import VDNMixer
import torch
from torch.optim import Adam
from utils.rl_utils import build_q_lambda_targets, build_td_lambda_targets


class QLearner:
    def __init__(self, args, mac, queue_center=None):
        self.args = args
        self.mac = mac
        self.queue_center = queue_center
        self.device = args.train_device
        self.train_batch_size = args.train_batch_size

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            if args.mixer == "qmix":
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)
        self.optimiser = Adam(params=self.params, lr=args.lr)
        self.target_mac = copy.deepcopy(mac)
        self.cuda()

    def train(self, batch, episode_num):
        max_length = batch["length"]
        obs = batch["obs"]
        state = batch["state"]
        actions = batch["actions"][:, :-1]
        actions_onehot = batch["actions_onehot"]
        avail_actions = batch["avail_actions"]
        rewards = batch["reward"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        mac_out = []
        hidden_states = self.mac.init_hidden(self.train_batch_size)
        for t in range(max_length):
            inputs = {}
            inputs["hidden_states"] = hidden_states
            inputs["obs"] = obs[:, t]
            inputs["avail_actions"] = avail_actions[:, t]
            if self.args.obs_last_action:
                if t == 0:
                    inputs["actions_onehot"] = torch.zeros_like(inputs["avail_actions"])
                else:
                    inputs["actions_onehot"] = actions_onehot[:, t - 1]
            agent_outs, hidden_states = self.mac.inference(inputs)
            mac_out.append(agent_outs)

        mac_out = torch.stack(mac_out, dim=1)
        chosen_action_qvals = torch.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        with torch.no_grad():
            target_mac_out = []
            hidden_states = self.target_mac.init_hidden(self.train_batch_size)
            for t in range(max_length):
                inputs = {}
                inputs["hidden_states"] = hidden_states
                inputs["obs"] = obs[:, t]
                inputs["avail_actions"] = avail_actions[:, t]
                if self.args.obs_last_action:
                    if t == 0:
                        inputs["actions_onehot"] = torch.zeros_like(inputs["avail_actions"])
                    else:
                        inputs["actions_onehot"] = actions_onehot[:, t - 1]
                agent_outs, hidden_states = self.target_mac.inference(inputs)
                target_mac_out.append(agent_outs)
            target_mac_out = torch.stack(target_mac_out, dim=1)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach.max(dim=3, keepdim=True)[1]
            target_max_qvals = torch.gather(target_mac_out, 3, cur_max_actions).squeeze(3)

            target_max_qvals = self.target_mixer(target_max_qvals, state)

            if getattr(self.args, 'q_lambda', False):
                qvals = torch.gather(target_mac_out, 3, batch["actions"]).squeeze(3)
                qvals = self.target_mixer(qvals, state)

                targets = build_q_lambda_targets(rewards, terminated, mask, target_max_qvals, qvals,
                                                 self.args.gamma, self.args.td_lambda)
            else:
                targets = build_td_lambda_targets(rewards, terminated, mask, target_max_qvals,
                                                  self.args.n_agents, self.args.gamma, self.args.td_lambda)

        chosen_action_qvals = self.mixer(chosen_action_qvals, state[:, :-1])

        td_error = (chosen_action_qvals - targets.detach())
        td_error2 = 0.5 * td_error.pow(2)

        mask = mask.expand_as(td_error2)
        masked_td_error = td_error2 * mask

        loss = masked_td_error.sum() / mask.sum()

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()
        self.mac.agent_version += 1

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

    def _update_targets(self):
        self.target_mac.load_agent_state(self.mac.agent_state())
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())

    def cuda(self):
        self.mac.to(self.device)
        self.target_mac.to(self.device)
        if self.mixer is not None:
            self.mixer.to(self.device)
            self.target_mixer.to(self.device)

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            torch.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        torch.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(
                torch.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(torch.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
