import torch
import torch.nn as nn
import torch.nn.functional as F
from policy.base.base_net import BaseAgent
from utils.utils import orthogonal_init_

agent_REGISTRY = {}
critic_REGISTRY = {}

class DDPGMlpAgent(BaseAgent):
    def __init__(self, args):
        super().__init__(args)
        self.fc1 = nn.Linear(self.input_shape, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, args.n_actions)
        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm(args.hidden_dim)
        if self.use_orthogonal:
            orthogonal_init_(self.fc1)
            orthogonal_init_(self.fc2, gain=args.gain)
            orthogonal_init_(self.fc3, gain=args.gain)
    
    def forward(self, data):
        inputs = self._build_inputs(data)
        b, a, _ = inputs.shape
        inputs = inputs.view(b * a, -1)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        x = F.tanh(self.fc3(x))
        return x.view(b, a, -1), None
    
class DDPGRnnAgent(BaseAgent):
    def __init__(self, args):
        super().__init__(args)
        self.fc1 = nn.Linear(self.input_shape, args.hidden_dim)
        self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.n_actions)
        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm(args.hidden_dim)    
        if self.use_orthogonal:
            orthogonal_init_(self.fc1)
            orthogonal_init_(self.fc2, gain=args.gain)
    
    def forward(self, data):
        hidden_states = data["hidden_states"].reshape(-1, self.args.hidden_dim)
        inputs = self._build_inputs(data)
        b, a, _ = inputs.shape
        inputs = inputs.view(b * a, -1)
        x = F.relu(self.fc1(inputs), inplace=True)
        hh = self.rnn(x, hidden_states)
        x = self.layer_norm(hh) if self.use_layer_norm else hh
        x = F.tanh(self.fc2(x))
        return x.view(b, a, -1), hh.view(b, a, -1)

agent_REGISTRY["ddpg_mlp"] = DDPGMlpAgent
agent_REGISTRY["ddpg_rnn"] = DDPGRnnAgent

class BaseCritic(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_layer_norm = args.use_layer_norm
        self.use_orthogonal = args.use_orthogonal
        self.input_shape = self._get_input_shape(args)
    
    def _get_input_shape(self, args):
        critic_input_shape = 0
        critic_input_shape += args.obs_shape * args.n_agents
        if "state" in self.args.critic_inputs:
            critic_input_shape += args.state_shape
        critic_input_shape += args.n_actions * args.n_agents
        return critic_input_shape
    
    def _build_inputs(self, data):
        inputs = []
        b, a, _ = data["obs"].shape
        device = data["obs"].device
        inputs.append(data["obs"].reshape(b, -1))
        inputs.append(data["actions"].reshape(b, -1))
        if "state" in self.args.critic_inputs:
            inputs.append(data["states"].reshape(b, -1))
        inputs = torch.cat([x.reshape(b, -1) for x in inputs], dim=-1).to(device)
        return inputs

class MlpCritic(BaseCritic):
    def __init__(self, args):
        super().__init__(args)
        self.fc1 = nn.Linear(self.input_shape, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, 1)
        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm(args.hidden_dim)
        if self.use_orthogonal:
            orthogonal_init_(self.fc1)
            orthogonal_init_(self.fc2, gain=args.gain)
    
    def forward(self, data):
        inputs = self._build_inputs(data)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(self.layer_norm(x))) if self.use_layer_norm else F.relu(self.fc2(x))
        v = self.fc3(x)
        return v, None
    
class RnnCritic(BaseCritic):
    def __init__(self, args):
        super().__init__(args)
        self.fc1 = nn.Linear(self.input_shape, args.hidden_dim)
        self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, 1)
        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm(args.hidden_dim)
        if self.use_orthogonal:
            orthogonal_init_(self.fc1)
            orthogonal_init_(self.fc2, gain=args.gain)
    
    def forward(self, data):
        b, a, _ = data["obs"].shape
        hidden_states = data["hidden_states"]
        inputs = self._build_inputs(data)
        x = F.relu(self.fc1(inputs), inplace=True)
        h_in = hidden_states.reshape(-1, self.args.hidden_dim)
        hh = self.rnn(x, h_in)
        x = self.layer_norm(hh) if self.use_layer_norm else hh
        v = self.fc2(x)
        return v, hh.view(b, a, -1)

critic_REGISTRY["mlp"] = MlpCritic
critic_REGISTRY["rnn"] = RnnCritic