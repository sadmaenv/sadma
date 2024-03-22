import torch

class BaseController:
    def __init__(self, args):
        self.args = args
        self.agent = None
        self.agent_version = torch.zeros(1, dtype=torch.int32)
        self.device = "cpu"
    
    def select_actions(self):
        pass
    
    def init_hidden_states(self, batch_size):
        return torch.zeros(batch_size, self.args.n_agents, self.args.hidden_dim).to(self.device)

    @property
    def agent_state_dict(self):
        return self.agent.state_dict()

    @property
    def agent_parameters(self):
        return list(self.agent.parameters())

    def load_agent_state_dict(self, agent_state_dict):
        self.agent.load_state_dict(agent_state_dict)

    def to(self, device):
        self.device = device
        self.agent.to(device)