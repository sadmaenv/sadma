import torch
from policy.base.base_controller import BaseController
from .q_net import agent_REGISTRY
from utils.utils import DecayThenFlatSchedule
from torch.distributions import Categorical

class QController(BaseController):
    def __init__(self, args):
        super().__init__(args)
        agent_type = f"q_{args.agent_type}"
        self.agent = agent_REGISTRY[agent_type](args)
        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time, decay="linear")
    
    def get_agent_outputs(self, data):
        data = {k: v.to(self.device) for k, v in data.items()}
        q, hidden_states = self.agent(data)
        return q, hidden_states
        
    def select_actions(self, data, steps, test_mode=False):
        data = {k: v.to(self.device) for k, v in data.items()}
        q, hidden_states = self.agent(data)
        q[data["available_actions"] == 0] = -float("inf")
        epsilon = 0 if test_mode else self.schedule.eval(steps)
        random_numbers = torch.rand_like(q[:, :, 0], device=self.device)
        pick_random = (random_numbers < epsilon).long()
        random_actions = Categorical(data["available_actions"].float()).sample().long()
        actions = (pick_random * random_actions + (1 - pick_random) * q.max(dim=2)[1])
        results = {"actions": actions.cpu()}
        if "rnn" in self.args.agent_type:
            results["hidden_states"] = hidden_states.cpu()
        return results

        
