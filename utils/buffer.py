import torch
import numpy as np

class BaseBuffer:
    def __init__(self, scheme, buffer_size, shared, device):
        self.device = device
        self.scheme = scheme
        self._setup_buffer(buffer_size, shared)

    def _setup_buffer(self, buffer_size, shared):
        self.datas = {}
        for field_key, field_info in self.scheme.items():
            vshape = field_info["vshape"]
            if isinstance(vshape, int):
                vshape = (vshape,)
            dtype = field_info.get("dtype", torch.float32)
            init_tensor = torch.zeros((buffer_size, *vshape), dtype=dtype).to(self.device)
            if shared and self.device == "cpu":
                init_tensor = init_tensor.share_memory_()
            self.datas[field_key] = init_tensor

class EpisodeBuffer(BaseBuffer):
    def __init__(self, args, scheme):
        self.args = args
        self.buffer_size = (args.num_sample_worker * args.env_batch_size * args.num_env_runner)
        if args.evaluate:
            self.buffer_size += args.evaluate_batch_size

        super(EpisodeBuffer, self).__init__(scheme, self.buffer_size, shared=True, device="cpu")

    def insert_data(self, data, bs, ts):
        for name, value in data.items():
            dtype = self.scheme[name].get("dtype", torch.float32)
            if not isinstance(value, torch.Tensor):
                value = torch.tensor(value, dtype=dtype)
            if name == "rewards" or name == "terminated":
                t = ts - 1
                t[t < 0] = 0
            else:
                t = ts
            self.datas[name][bs, t] = value
            if name == "obs":
                self.datas["filled"][bs, t] = 1

    def get_infer_data(self, bs, ts):
        infer_data = {}
        obs = self.datas["obs"][bs, ts]
        available_action = self.datas["available_actions"][bs, ts]
        infer_data = {"obs": obs, "available_actions":available_action}
        pre_ts = ts - 1
        pre_index = np.nonzero(pre_ts >= 0)[0]
        pre_index = pre_index.tolist()
        if "rnn" in self.args.agent_type:
            hidden_state = torch.zeros_like(self.datas["hidden_states"][bs, ts])
            if pre_index:
                hidden_state[pre_index] = self.datas["hidden_states"][bs[pre_index], pre_ts[pre_index]]
            infer_data["hidden_states"] = hidden_state
        if "last_action" in self.args.agent_inputs:
            last_action = torch.zeros_like(available_action)
            if pre_index:
                action_onehot = self.datas["actions"][bs[pre_index], pre_ts[pre_index]].squeeze(-1)
                action_onehot = torch.nn.functional.one_hot(action_onehot, self.args.n_actions)
                action_onehot = action_onehot.to(torch.int32)
                last_action[pre_index] = action_onehot
            infer_data["last_actions"] = last_action
        return infer_data
    
    def reset_data(self, bs):
        for name, value in self.datas.items():
            value[bs] = 0


    def __getitem__(self, item):
        if isinstance(item, list) or isinstance(item, np.ndarray):
            episodes = {}
            for name, value in self.datas.items():
                episodes[name] = value[item].clone()
                value[item] = 0
            return episodes
        elif isinstance(item, str):
            return self.datas[item]
        elif isinstance(item, tuple):
            data_name, bs, ts = item
            if (data_name == "hidden_states"or data_name == "terminated"):
                t = ts - 1
                t[t < 0] = 0
            else:
                t = ts
            result_data = self.datas[data_name][bs, t]
            return result_data

class SharedBuffer(BaseBuffer):
    def __init__(self, args, scheme):
        self.buffer_size = args.shared_buffer_size
        self.scheme = scheme
        super(SharedBuffer, self).__init__(scheme, self.buffer_size, shared=True, device="cpu")

    def insert_episode(self, buffer_idx, episodes_data):
        for k, v in episodes_data.items():
            self.datas[k][buffer_idx] = v

    def __getitem__(self, buffer_idx):
        datas = {}
        for k, v in self.datas.items():
            datas[k] = v[buffer_idx]
        return datas


class ReplayBuffer(BaseBuffer):
    def __init__(self, args, scheme):
        self.buffer_size = args.train_buffer_size
        super(ReplayBuffer, self).__init__(scheme, self.buffer_size, False, "cpu")
        self.index_count = 0
        self.episodes_in_buffer = 0
        self.train_batch_size = args.train_batch_size
        self.train_device = args.train_device

    def can_sample(self):
        return self.episodes_in_buffer >= self.train_batch_size

    def insert_episodes(self, batch_size, episodes):
        if self.index_count + batch_size <= self.buffer_size:
            target_slice = slice(self.index_count, self.index_count + batch_size)
            for name, value in episodes.items():
                self.datas[name][target_slice] = value
            self.index_count = self.index_count + batch_size
            self.episodes_in_buffer = max(self.episodes_in_buffer, self.index_count)
            self.index_count = self.index_count % self.buffer_size
        else:
            left = self.buffer_size - self.index_count
            right = batch_size - left
            for name, value in episodes.items():
                self.datas[name][self.index_count :] = value[:left]
                self.datas[name][:right] = value[left:]
            self.index_count += batch_size
            self.episodes_in_buffer = self.buffer_size
            self.index_count = self.index_count % self.buffer_size
    
    def sample(self):
        batch_episodes = {}
        index = np.random.choice(self.episodes_in_buffer, self.train_batch_size, replace=False)
        for name, value in self.datas.items():
            batch_episodes[name] = value[index]
        max_length = torch.sum(batch_episodes["filled"], 1).max(0)[0]
        for name, value in batch_episodes.items():
            batch_episodes[name] = value[:, :max_length].to(self.train_device)
        return batch_episodes