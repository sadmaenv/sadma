import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from policy.base.base_net import BaseAgent
from utils.utils import orthogonal_init_


agent_REGISTRY = {}
mixer_REGISTRY = {}

class QMlpAgent(BaseAgent):
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
        x = self.layer_norm(x) if self.use_layer_norm else x
        q = self.fc3(x)
        return q.view(b, a, -1), None
    
class QRnnAgent(BaseAgent):
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
        q = self.fc2(x)
        return q.view(b, a, -1), hh.view(b, a, -1)
    
agent_REGISTRY["q_mlp"] = QMlpAgent
agent_REGISTRY["q_rnn"] = QRnnAgent

class VDNMixer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
    
    def forward(self, qs, state):
        return torch.sum(qs, dim=2, keepdim=True)
    
class Qmixer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.embed_dim = args.mixing_embed_dim
        self.abs = getattr(self.args, 'abs', True)

        if getattr(args, "hypernet_layers", 1) == 1:
            self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
            self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
        elif getattr(args, "hypernet_layers", 1) == 2:
            hypernet_embed = self.args.hypernet_embed
            self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(hypernet_embed, self.embed_dim * self.n_agents))
            self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                               nn.ReLU(inplace=True),
                                               nn.Linear(hypernet_embed, self.embed_dim))
        elif getattr(args, "hypernet_layers", 1) > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")

        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(inplace=True),
                               nn.Linear(self.embed_dim, 1))
    
    def forward(self, qs, states):
        b, a, _ = qs.shape
        states = states.reshape(b * a, -1)
        qs = qs.reshape(b * a, 1, -1)
        # First layer
        w1 = self.hyper_w_1(states).abs() if self.abs else self.hyper_w_1(states)
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(qs, w1) + b1)
        # Second layer
        w_final = self.hyper_w_final(states).abs() if self.abs else self.hyper_w_final(states)
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)
        # Compute final output
        y = torch.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(b, -1, 1)
        return q_tot
    
mixer_REGISTRY["vdn_mixer"] = VDNMixer
mixer_REGISTRY["q_mixer"] = Qmixer