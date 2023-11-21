import torch as th
import torch.nn as nn
import torch.nn.functional as F
from utils.torch_utils import orthogonal_init_
from torch.nn import LayerNorm


class CentralVMLPCritic(nn.Module):
    def __init__(self, args):
        super(CentralVMLPCritic, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.device = args.train_device

        input_shape = self._get_input_shape(args)
        self.output_type = "v"

        # Set up network layers
        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, 1)

    def forward(self, data):
        inputs = self._build_inputs(data)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q

    def _build_inputs(self, data):
        b, max_t, _ = data["state"].size()

        inputs = []
        # state
        inputs.append(data["state"].unsqueeze(2).repeat(1, 1, self.n_agents, 1))

        # observations
        if self.args.obs_individual_obs:
            inputs.append(data["obs"][:].view(b, max_t, -1).unsqueeze(2).repeat(1, 1, self.n_agents, 1))

        # last actions
        if self.args.obs_last_action:
            last_actions = th.cat([th.zeros_like(data["actions_onehot"][:, 0:1]), data["actions_onehot"][:, :-1]], dim=1)
            last_actions = last_actions.view(b, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
            inputs.append(last_actions)

        inputs.append(th.eye(self.n_agents, device=self.device).unsqueeze(0).unsqueeze(0).expand(b, max_t, -1, -1))

        inputs = th.cat(inputs, dim=-1)
        return inputs

    def _get_input_shape(self, args):
        # state
        input_shape = args.state_shape
        # observations
        if self.args.obs_individual_obs:
            input_shape += args.obs_shape * self.n_agents
        # last actions
        if self.args.obs_last_action:
            input_shape += args.n_actions * self.n_agents
        input_shape += self.n_agents
        return input_shape


class CentralVRNNCritic(nn.Module):
    def __init__(self, args):
        super(CentralVRNNCritic, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.device = args.train_device

        input_shape = self._get_input_shape(args)
        self.output_type = "v"

        # Set up network layers
        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, 1)

        if getattr(args, "use_layer_norm", False):
            self.layer_norm = LayerNorm(args.hidden_dim)

        if getattr(args, "use_orthogonal", False):
            orthogonal_init_(self.fc1)
            orthogonal_init_(self.fc2, gain=args.gain)

    def forward(self, data):
        inputs = self._build_inputs(data)
        b, max_t, e = inputs.size()
        inputs = inputs.view(-1, e)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q.view(b, max_t, -1)

    def _build_inputs(self, data):
        b, max_t, _ = data["state"].size()

        inputs = []
        # state
        inputs.append(data["state"].unsqueeze(2).repeat(1, 1, self.n_agents, 1))

        # observations
        if self.args.obs_individual_obs:
            inputs.append(data["obs"][:].view(b, max_t, -1).unsqueeze(2).repeat(1, 1, self.n_agents, 1))

        # last actions
        if self.args.obs_last_action:
            last_actions = th.cat([th.zeros_like(data["actions_onehot"][:, 0:1]), data["actions_onehot"][:, :-1]], dim=1)
            last_actions = last_actions.view(b, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
            inputs.append(last_actions)

        inputs.append(th.eye(self.n_agents, device=self.device).unsqueeze(0).unsqueeze(0).expand(b, max_t, -1, -1))

        inputs = th.cat(inputs, dim=-1)
        return inputs

    def _get_input_shape(self, args):
        # state
        input_shape = args.state_shape
        # observations
        if self.args.obs_individual_obs:
            input_shape += args.obs_shape * self.n_agents
        # last actions
        if self.args.obs_last_action:
            input_shape += args.n_actions * self.n_agents
        input_shape += self.n_agents
        return input_shape
