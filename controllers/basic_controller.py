import torch
from utils.action_selectors import REGISTRY as action_REGISTRY
from modules.agents import REGISTRY as agent_REGISTRY
from utils.utils import one_hot


class BasicMAC:
    def __init__(self, args):
        self.args = args

        self.hidden_states = None
        self.action_selector = action_REGISTRY[args.action_selector](args)
        self.agent = agent_REGISTRY[self.args.agent](args)
        self.agent_version = torch.zeros(1, dtype=torch.int32)
        self.device = "cpu"

    def select_actions(self, data, total_step=0, test_mode=False):
        avail_actions = data["avail_actions"]
        agent_outs, hidden_states = self.inference(data)
        agent_outs = agent_outs.cpu()
        result = {}
        if self.args.save_probs:
            actions, probs = self.action_selector.select_action(agent_outs, avail_actions, total_step, test_mode)
            result["probs"] = probs.cpu()
        else:
            actions = self.action_selector.select_action(agent_outs, avail_actions, total_step, test_mode)
        result["actions"] = actions.unsqueeze(-1)
        result["actions_onehot"] = one_hot(actions, self.args.n_actions)
        if "rnn" in self.args.agent:
            result["hidden_states"] = hidden_states.detach().cpu()
        return result

    def inference(self, data):
        avail_actions = data["avail_actions"]
        batch_size, n_agents, _ = avail_actions.size()
        agent_inputs = self._build_inputs(data)
        if "rnn" in self.args.agent:
            hidden_states = data["hidden_states"].to(self.device)
            agent_outs, hidden_states = self.agent(agent_inputs, hidden_states)
        else:
            agent_outs = self.agent(agent_inputs)
            hidden_states = None

        if self.args.agent_output_type == "pi_logits":
            if getattr(self.args, "mask_before_softmax", True):
                agent_outs = agent_outs.reshape(batch_size * n_agents, -1)
                reshaped_avail_actions = avail_actions.reshape(batch_size * n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e5
            agent_outs = torch.nn.functional.softmax(agent_outs, dim=-1)
        agent_outs = agent_outs.view(batch_size, n_agents, -1)

        return agent_outs, hidden_states

    def _build_inputs(self, data):
        inputs = []
        obs = data["obs"].to(self.device)
        b, a, _ = obs.size()
        inputs.append(obs)
        if self.args.obs_last_action:
            last_actions = data["actions_onehot"].to(self.device)
            inputs.append(last_actions)
        if self.args.obs_agent_id:
            agent_id_tensor = torch.eye(a, device=self.device).unsqueeze(0).expand(b, -1, -1)
            inputs.append(agent_id_tensor)
        inputs = torch.cat([x.reshape(b, a, -1) for x in inputs], dim=-1).to(self.device)
        return inputs

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.args.n_agents, self.args.hidden_dim).to(self.device)

    def agent_state(self):
        return self.agent.state_dict()

    def parameters(self):
        return self.agent.parameters()

    def load_agent_state(self, agent_state_dict):
        self.agent.load_state_dict(agent_state_dict)

    def to(self, device):
        self.device = device
        self.agent.to(device)
