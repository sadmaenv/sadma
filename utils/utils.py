import os
import torch
from torch import nn
import numpy as np


# --- net utils ---
def get_parameters_num(param_list):
    return str(sum(p.numel() for p in param_list) / 1000) + 'K'

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

def orthogonal_init_(m, gain=1):
    if isinstance(m, nn.Linear):
        init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)


# --- rl utils ---
def build_td_lambda_targets(rewards, terminated, mask, target_qs, n_agents, gamma, td_lambda):
    # Assumes  <target_qs > in B*T*A and <reward >, <terminated >, <mask > in (at least) B*T-1*1
    # Initialise  last  lambda -return  for  not  terminated  episodes
    ret = target_qs.new_zeros(*target_qs.shape)
    ret[:, -1] = target_qs[:, -1] * (1 - torch.sum(terminated, dim=1))
    # Backwards  recursive  update  of the "forward  view"
    for t in range(ret.shape[1] - 2, -1, -1):
        ret[:, t] = td_lambda * gamma * ret[:, t + 1] + mask[:, t] * (
                rewards[:, t]
                + (1 - td_lambda) * gamma * target_qs[:, t + 1] * (1 - terminated[:, t])
        )
    # Returns lambda-return from t=0 to t=T-1, i.e. in B*T-1*A
    return ret[:, 0:-1]


def build_gae_targets(rewards, masks, values, gamma, lambd):
    B, T, A, _ = values.size()
    T -= 1
    advantages = torch.zeros(B, T, A, 1).to(device=values.device)
    advantage_t = torch.zeros(B, A, 1).to(device=values.device)

    for t in reversed(range(T)):
        delta = rewards[:, t] + values[:, t + 1] * gamma * masks[:, t] - values[:, t]
        advantage_t = delta + advantage_t * gamma * lambd * masks[:, t]
        advantages[:, t] = advantage_t

    returns = values[:, :T] + advantages
    return advantages, returns

def build_vtrace_gae_targets(ratio, rewards, masks, values, gamma, lambd):
    B, T, A, _ = values.size()
    T -= 1
    c = torch.min(torch.FloatTensor([1.0]).to(ratio.device), ratio)
    p = torch.min(torch.FloatTensor([1.0]).to(ratio.device), ratio)

    advantages = torch.zeros(B, T, A, 1).to(device=values.device)
    advantage_t = torch.zeros(B, A, 1).to(device=values.device)

    for t in reversed(range(T)):
        delta = (
                        rewards[:, t] + values[:, t + 1] * gamma * masks[:, t] - values[:, t]
                ) * p[:, t]
        advantage_t = delta + advantage_t * gamma * lambd * masks[:, t] * c[:, t]
        advantages[:, t] = advantage_t

    returns = values[:, :T] + advantages
    return advantages, returns

def build_q_lambda_targets(rewards, terminated, mask, exp_qvals, qvals, gamma, td_lambda):
    # Assumes  <target_qs > in B*T*A and <reward >, <terminated >, <mask > in (at least) B*T-1*1
    # Initialise  last  lambda -return  for  not  terminated  episodes
    ret = exp_qvals.new_zeros(*exp_qvals.shape)
    ret[:, -1] = exp_qvals[:, -1] * (1 - torch.sum(terminated, dim=1))
    # Backwards  recursive  update  of the "forward  view"
    for t in range(ret.shape[1] - 2, -1, -1):
        reward = rewards[:, t] + exp_qvals[:, t] - qvals[:, t]  # off-policy correction
        ret[:, t] = td_lambda * gamma * ret[:, t + 1] + mask[:, t] * (
                reward
                + (1 - td_lambda) * gamma * exp_qvals[:, t + 1] * (1 - terminated[:, t])
        )
    # Returns lambda-return from t=0 to t=T-1, i.e. in B*T-1*A
    return ret[:, 0:-1]

class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = (
                m_a
                + m_b
                + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        )
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

class ValueNorm(nn.Module):
    """Normalize a vector of observations - across the first norm_axes dimensions"""

    def __init__(
            self,
            input_shape,
            norm_axes=1,
            beta=0.99999,
            per_element_update=False,
            epsilon=1e-5,
            device=torch.device("cpu"),
    ):
        super(ValueNorm, self).__init__()

        self.input_shape = input_shape
        self.norm_axes = norm_axes
        self.epsilon = epsilon
        self.beta = beta
        self.per_element_update = per_element_update
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.running_mean = nn.Parameter(torch.zeros(input_shape), requires_grad=False).to(**self.tpdv)
        self.running_mean_sq = nn.Parameter(torch.zeros(input_shape), requires_grad=False).to(**self.tpdv)
        self.debiasing_term = nn.Parameter(torch.tensor(0.0), requires_grad=False).to(**self.tpdv)

        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_mean_sq.zero_()
        self.debiasing_term.zero_()

    def running_mean_var(self):
        debiased_mean = self.running_mean / self.debiasing_term.clamp(min=self.epsilon)
        debiased_mean_sq = self.running_mean_sq / self.debiasing_term.clamp(
            min=self.epsilon
        )
        debiased_var = (debiased_mean_sq - debiased_mean ** 2).clamp(min=1e-2)
        return debiased_mean, debiased_var

    @torch.no_grad()
    def update(self, input_vector):
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)

        batch_mean = input_vector.mean(dim=tuple(range(self.norm_axes)))
        batch_sq_mean = (input_vector ** 2).mean(dim=tuple(range(self.norm_axes)))

        if self.per_element_update:
            batch_size = np.prod(input_vector.size()[: self.norm_axes])
            weight = self.beta ** batch_size
        else:
            weight = self.beta

        self.running_mean.mul_(weight).add_(batch_mean * (1.0 - weight))
        self.running_mean_sq.mul_(weight).add_(batch_sq_mean * (1.0 - weight))
        self.debiasing_term.mul_(weight).add_(1.0 * (1.0 - weight))

    def normalize(self, input_vector):
        # Make sure input is float32
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)

        mean, var = self.running_mean_var()
        out = (input_vector - mean[(None,) * self.norm_axes]) / torch.sqrt(var)[
            (None,) * self.norm_axes
            ]

        return out

    def denormalize(self, input_vector):
        """Transform normalized data back into original distribution"""
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)

        mean, var = self.running_mean_var()
        out = (
                input_vector * torch.sqrt(var)[(None,) * self.norm_axes]
                + mean[(None,) * self.norm_axes]
        )

        return out

class DecayThenFlatSchedule():

    def __init__(self,
                 start,
                 finish,
                 time_length,
                 decay="exp"):

        self.start = start
        self.finish = finish
        self.time_length = time_length
        self.delta = (self.start - self.finish) / self.time_length
        self.decay = decay

        if self.decay in ["exp"]:
            self.exp_scaling = (-1) * self.time_length / np.log(self.finish) if self.finish > 0 else 1

    def eval(self, T):
        if self.decay in ["linear"]:
            return max(self.finish, self.start - self.delta * T)
        elif self.decay in ["exp"]:
            return min(self.start, max(self.finish, np.exp(- T / self.exp_scaling)))


class LinearIncreaseSchedule():

    def __init__(self,
                 start,
                 finish,
                 time_length):
        self.start = start
        self.finish = finish
        self.time_length = time_length
        self.delta = (self.start - self.finish) / self.time_length

    def eval(self, T):
        return min(self.finish, self.start - self.delta * T)
    
    
# --- type utils ---
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

# --- other utils ---
def mkdir(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)