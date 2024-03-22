import torch
from policy.base.base_controller import BaseController
from .ppo_net import agent_REGISTRY


class PPOController(BaseController):
    def __init__(self, args):
        super().__init__(args)
        agent_type = f"pi_{args.agent_type}"
        self.agent = agent_REGISTRY[agent_type](args)
    
    def make_distribution(self, data, actions):
        data = {k: v.to(self.device) for k, v in data.items()}
        action_distribution, hidden_states = self.agent(data)
        action_log_probs = action_distribution.log_prob(actions)
        entropy = action_distribution.entropy()
        return action_log_probs, entropy, hidden_states
    
    def select_actions(self, data, steps, test_mode=False):
        data = {k: v.to(self.device) for k, v in data.items()}
        action_distribution, hidden_states = self.agent(data)
        actions = action_distribution.mode if test_mode else action_distribution.sample()
        action_log_probs = action_distribution.log_prob(actions)            
        results = {
                "actions": actions.cpu(), 
                "action_log_probs": action_log_probs.cpu(), 
                }
        if "rnn" in self.args.agent_type:
            results["hidden_states"] = hidden_states.cpu()
        return results
