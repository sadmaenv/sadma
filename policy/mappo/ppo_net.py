import torch
import torch.nn as nn
import torch.nn.functional as F
from policy.base.base_net import BaseAgent
from utils.utils import orthogonal_init_
from torch.distributions import Categorical, Normal

agent_REGISTRY = {}
critic_REGISTRY = {}

class PiMlpAgent(BaseAgent):
    def __init__(self, args):
        super(PiMlpAgent, self).__init__(args)
        self.fc1 = nn.Linear(self.input_shape, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, args.n_actions)
        if args.env_type == "continuous":
            self.log_std = nn.Parameter(torch.zeros(1, args.n_actions))
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
        x = self.fc3(x)
        x = x.view(b, a, -1)
        if self.env_type == "discrete":
            available_actions = data["available_actions"].reshape(b * a, -1)
            x[available_actions==0] = -1e10
            return Categorical(logits=x), None
        else:
            log_std = self.log_std.expand_as(x)
            std = torch.exp(log_std)
            return Normal(x, std), None

class PiRnnAgent(BaseAgent):
    def __init__(self, args):
        super(PiRnnAgent, self).__init__(args)
        self.fc1 = nn.Linear(self.input_shape, args.hidden_dim)
        self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.n_actions)
        if args.env_type == "continuous":
            self.log_std = nn.Parameter(torch.zeros(1, args.n_actions))
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
        x = self.fc2(x)
        if self.env_type == "discrete":
            available_actions = data["available_actions"].reshape(b * a, -1)
            x[available_actions==0] = -1e10
            x = x.reshape(b, a, -1)
            return Categorical(logits=x), hh.view(b, a, -1)
        else:
            x = x.reshape(b, a, -1)
            log_std = self.log_std.expand_as(x)
            std = torch.exp(log_std)
            return Normal(x, std), hh.view(b, a, -1)

agent_REGISTRY["pi_mlp"] = PiMlpAgent
agent_REGISTRY["pi_rnn"] = PiRnnAgent


class BaseCritic(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_layer_norm = args.use_layer_norm
        self.use_orthogonal = args.use_orthogonal
        self.input_shape = self._get_input_shape(args)

    def _get_input_shape(self, args):
        critic_input_shape = 0
        # ippo
        if "independent_obs" in self.args.critic_inputs:
            critic_input_shape += args.obs_shape
            if "last_action" in self.args.critic_inputs:
                critic_input_shape += args.n_actions
            if "agent_id" in self.args.critic_inputs:
                critic_input_shape += args.n_agents
        # mappo
        elif "central_obs" in self.args.critic_inputs:
            critic_input_shape += args.obs_shape * args.n_agents
            if "state" in self.args.critic_inputs:
                critic_input_shape += args.state_shape
        return critic_input_shape

    def _build_inputs(self, data):
        inputs = []
        b, a, _ = data["obs"].shape
        device = data["obs"].device
        if "independent_obs" in self.args.critic_inputs:
            inputs.append(data["obs"])
            if "last_action" in self.args.critic_inputs:
                inputs.append(data["last_actions"])
            if "agent_id" in self.args.critic_inputs:
                agent_id = torch.eye(a, device=device).unsqueeze(0).expand(b, -1, -1) 
                inputs.append(agent_id) 
            inputs = torch.cat([x.reshape(b, a, -1) for x in inputs], dim=-1).to(device)
            inputs = inputs.view(b * a, -1)
        elif "central_obs" in self.args.critic_inputs:
            inputs.append(data["obs"].reshape(b, -1))
            if "state" in self.args.critic_inputs:
                inputs.append(data["states"].reshape(b, -1))       
            inputs = torch.cat([x.reshape(b, -1) for x in inputs], dim=-1).to(device)
        return inputs

class MlpCritic(BaseCritic):
    def __init__(self, args):
        super(MlpCritic, self).__init__(args)
        self.fc1 = nn.Linear(self.input_shape, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, 1)
        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm(args.hidden_dim)
        if self.use_orthogonal:
            orthogonal_init_(self.fc1)
            orthogonal_init_(self.fc2, gain=args.gain)

    def forward(self, data):
        b, a, _ = data["obs"].shape
        inputs = self._build_inputs(data)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(self.layer_norm(x))) if self.use_layer_norm else F.relu(self.fc2(x))
        v = self.fc3(x)
        if torch.numel(v) < b * a:
            v = v.repeat(1, a).unsqueeze(-1)
        return v.view(b, a, -1), None

class RnnCritic(BaseCritic):
    def __init__(self, args):
        super(RnnCritic, self).__init__(args)
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
        if "central_obs" in self.args.critic_inputs:
            v = v.unsqueeze(1).expand(-1, a, -1)
            hh = hh.view(b, -1)
        elif "independent_obs" in self.args.critic_inputs:
            v = v.view(b, a, -1)
            hh = hh.view(b, a, -1)
        return v, hh

critic_REGISTRY["mlp"] = MlpCritic
critic_REGISTRY["rnn"] = RnnCritic
