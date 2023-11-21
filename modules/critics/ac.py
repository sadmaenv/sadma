import torch.nn as nn
import torch.nn.functional as F
from utils.torch_utils import orthogonal_init_
from torch.nn import LayerNorm
import torch


class ACMLPCritic(nn.Module):
    def __init__(self, args):
        super(ACMLPCritic, self).__init__()
        self.args = args
        self.device = args.train_device
        self.train_batch_size = args.train_batch_size

        self.n_actions = args.n_actions
        self.n_agents = args.n_agents

        input_shape = self._get_input_shape(args)

        # Set up network layers
        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, 1)

        if getattr(args, "use_layer_norm", False):
            self.layer_norm = LayerNorm(args.hidden_dim)

        if getattr(args, "use_orthogonal", False):
            orthogonal_init_(self.fc1)
            orthogonal_init_(self.fc2, gain=args.gain)

    def forward(self, data):
        inputs = self._build_inputs(data)
        x = F.relu(self.fc1(inputs))
        if getattr(self.args, "use_layer_norm", False):
            x = F.relu(self.fc2(self.layer_norm(x)))
        else:
            x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q

    def _build_inputs(self, data):
        max_t = data["length"]
        inputs = []
        # observations
        inputs.append(data["obs"][:])
        inputs.append(torch.eye(self.n_agents, device=self.device).unsqueeze(0).unsqueeze(0).expand(self.train_batch_size, max_t, -1, -1))
        inputs = torch.cat(inputs, dim=-1).to(self.device)
        return inputs

    def _get_input_shape(self, args):
        # observations
        input_shape = args.obs_shape
        # agent id
        input_shape += self.n_agents
        return input_shape


class ACRNNCritic(nn.Module):
    def __init__(self, args):
        super(ACRNNCritic, self).__init__()
        self.args = args
        self.hidden_dim = args.hidden_dim

        input_shape = self._get_input_shape(args)
        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, 1)

        if getattr(args, "use_layer_norm", False):
            self.layer_norm = LayerNorm(args.hidden_dim)

        if getattr(args, "use_orthogonal", False):
            orthogonal_init_(self.fc1)
            orthogonal_init_(self.fc2, gain=args.gain)

        self.hidden_states = None

    def _get_input_shape(self, args):
        agent_input_shape = args.obs_shape
        if args.obs_last_action:
            agent_input_shape += args.n_actions
        if args.obs_agent_id:
            agent_input_shape += args.n_agents
        return agent_input_shape

    def init_hidden(self, batch_size):
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_().unsqueeze(0).expand(batch_size, self.args.n_agents, -1)

    def set_hidden(self, hidden_states):
        self.hidden_states = hidden_states.to(self.fc1.weight.device)

    def forward(self, data, hidden_states):
        inputs = self._build_inputs(data)
        b, a, e = inputs.size()
        inputs = inputs.view(-1, e)
        x = F.relu(self.fc1(inputs))
        h_in = hidden_states.reshape(-1, self.hidden_dim)
        h = self.rnn(x, h_in)
        # self.hidden_states = h
        if getattr(self.args, "use_layer_norm", False):
            q = self.fc2(self.layer_norm(h))
        else:
            q = self.fc2(h)
        return q.view(b, a, -1), h.view(b, a, -1)

    def _build_inputs(self, data):
        device = self.fc1.weight.device
        inputs = []
        obs = data["obs"].to(device)
        b, a, _ = obs.size()
        inputs.append(obs)
        if self.args.obs_last_action:
            last_actions = data["actions_onehot"].to(device)
            inputs.append(last_actions)
        if self.args.obs_agent_id:
            agent_id_tensor = torch.eye(a, device=device).unsqueeze(0).expand(b, -1, -1)
            inputs.append(agent_id_tensor)
        inputs = torch.cat([x.reshape(b, a, -1) for x in inputs], dim=-1).to(device)
        return inputs
