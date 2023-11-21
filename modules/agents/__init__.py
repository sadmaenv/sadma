REGISTRY = {}

from .rnn_agent import RNNAgent
from .mlp_agent import MLPAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["mlp"] = MLPAgent
