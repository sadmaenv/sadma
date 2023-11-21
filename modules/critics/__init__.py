from .ac import ACMLPCritic, ACRNNCritic
from .centralV import CentralVMLPCritic, CentralVRNNCritic

REGISTRY = {}

REGISTRY["ac_mlp_critic"] = ACMLPCritic
REGISTRY["ac_rnn_critic"] = ACRNNCritic
REGISTRY["cv_mlp_critic"] = CentralVMLPCritic
REGISTRY["cv_rnn_critic"] = CentralVRNNCritic
