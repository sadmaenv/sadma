import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseAgent(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.env_type = args.env_type
        self.use_layer_norm = args.use_layer_norm
        self.use_orthogonal = args.use_orthogonal
        self.input_shape = self._get_input_shape(args)

    def _get_input_shape(self, args):
        agent_input_shape = 0
        if "obs" in self.args.agent_inputs:
            agent_input_shape += args.obs_shape
        if "last_action" in self.args.agent_inputs:
            agent_input_shape += args.n_actions
        if "agent_id" in self.args.agent_inputs:
            agent_input_shape += args.n_agents
        return agent_input_shape

    def _build_inputs(self, data):
        inputs = []
        b, a, _ = data["obs"].shape
        device = data["obs"].device
        if "obs" in self.args.agent_inputs:
            inputs.append(data["obs"])
        if "last_action" in self.args.agent_inputs:
            inputs.append(data["last_actions"])
        if "agent_id" in self.args.agent_inputs:
            agent_id = torch.eye(a, device=device).unsqueeze(0).expand(b, -1, -1)
            inputs.append(agent_id)
        inputs = torch.cat([x.reshape(b, a, -1) for x in inputs], dim=-1).to(device)
        return inputs