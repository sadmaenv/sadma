import torch.nn as nn
import torch.nn.functional as F


class MLPAgent(nn.Module):
    def __init__(self, args):
        super(MLPAgent, self).__init__()
        self.args = args
        input_shape = self._get_input_shape(args)
        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, args.n_actions)

        self.agent_return_logits = getattr(self.args, "agent_return_logits", False)

    def _get_input_shape(self, args):
        agent_input_shape = args.obs_shape
        if args.obs_last_action:
            agent_input_shape += args.n_actions
        if args.obs_agent_id:
            agent_input_shape += args.n_agents
        return agent_input_shape

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        if self.agent_return_logits:
            actions = self.fc3(x)
        else:
            actions = F.tanh(self.fc3(x))
        return actions
