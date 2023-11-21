import torch.nn as nn
import torch.nn.functional as F
from utils.torch_utils import orthogonal_init_
from torch.nn import LayerNorm


class RNNAgent(nn.Module):
    def __init__(self, args):
        super(RNNAgent, self).__init__()
        self.args = args
        input_shape = self._get_input_shape(args)
        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.n_actions)

        if getattr(args, "use_layer_norm", False):
            self.layer_norm = LayerNorm(args.hidden_dim)

        if getattr(args, "use_orthogonal", False):
            orthogonal_init_(self.fc1)
            orthogonal_init_(self.fc2, gain=args.gain)

    def _get_input_shape(self, args):
        agent_input_shape = args.obs_shape
        if args.obs_last_action:
            agent_input_shape += args.n_actions
        if args.obs_agent_id:
            agent_input_shape += args.n_agents
        return agent_input_shape

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        b, a, e = inputs.size()

        inputs = inputs.view(-1, e)
        x = F.relu(self.fc1(inputs), inplace=True)
        h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        hh = self.rnn(x, h_in)

        if getattr(self.args, "use_layer_norm", False):
            q = self.fc2(self.layer_norm(hh))
        else:
            q = self.fc2(hh)

        return q.view(b, a, -1), hh.view(b, a, -1)
