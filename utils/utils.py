import torch
import numpy as np


def one_hot(tensor, out_dim):
    return torch.nn.functional.one_hot(tensor, out_dim)


def get_numpy_type_name(numpy_type):
    n = np.zeros(1, numpy_type)
    return n.dtype.name


def to_numpy_type(torch_type):
    t = torch.zeros([1], dtype=torch_type)
    n = t.numpy()
    return n.dtype


def to_torch_type(numpy_type):
    n = np.zeros(1, dtype=numpy_type)
    t = torch.tensor(n)
    return t.dtype


def get_bs(actor_id, group_id, env_batch_size, num_env_runner):
    start = env_batch_size * (actor_id * num_env_runner + group_id)
    end = start + env_batch_size
    return np.arange(start, end)


def make_batch(episodes_data, device):
    max_length = torch.sum(episodes_data["filled"], 1).max(0)[0]
    for name, value in episodes_data.items():
        episodes_data[name] = value[:, :max_length].to(device)
    episodes_data["length"] = max_length
    return episodes_data
