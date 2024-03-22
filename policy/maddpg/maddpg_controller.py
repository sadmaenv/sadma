import torch
from policy.base.base_controller import BaseController
from .maddpg_net import agent_REGISTRY

class DDPGController(BaseController):
    def __init__(self, args):
        super().__init__(args)
        agent_type = f"ddpg_{args.agent_type}"
        self.agent = agent_REGISTRY[agent_type](args)
    
    def select_actions(self, data, steps, test_mode=False):
        data = {k: v.to(self.device) for k, v in data.items()}
        actions, hidden_states = self.agent(data)
        noisy = 0 if test_mode else torch.rand_like(actions)