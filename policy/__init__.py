from policy.mappo.ppo_controller import PPOController
from policy.mappo.ppo_policy import PPOPolicy
from policy.qmix.q_controller import QController
from policy.qmix.q_policy import Qpolicy

policy_REGISTRY = {}
policy_REGISTRY["ppo"] = {"mac": PPOController, "policy": PPOPolicy}
policy_REGISTRY["qmix"] = {"mac": QController, "policy": Qpolicy}
