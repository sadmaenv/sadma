import copy
import torch
import numpy as np
from worker.base_worker import ProcessWorker
from torch.multiprocessing import Process

class InferWorker(ProcessWorker):
    def __init__(self, args, mac, episode_buffer, queue_center, shared_buffer, steps):
        super().__init__()
        self.args = args
        self.mac = mac
        self.episode_buffer = episode_buffer
        self.env_batch_size = args.env_batch_size
        self.queue_center = queue_center
        self.shared_buffer = shared_buffer
        self.steps = steps

        self.job = Process(target=self.async_infer)

        self.batch_infer = {
            "infer_bs": [],
            "infer_ts": [],
            "sample_id": [],
            "alive_env": [],
        }

    def async_infer(self):
        torch.set_num_threads(1)
        device = self.args.train_device if self.args.infer_device == "cuda" else "cpu"
        local_mac = copy.deepcopy(self.mac)
        local_mac.to(device)
        self.queue_center.put("log", {"type": "info", "msg": "infer worker start"})
        while True:
            self.batch_infer = {key: [] for key in self.batch_infer}
            for _ in range(self.args.infer_batch_size):
                infer_info = self.queue_center.get("infer_request")
                for k, v in infer_info.items():
                    self.batch_infer[k].append(v)

            infer_bs = np.hstack(self.batch_infer["infer_bs"])
            infer_ts = np.hstack(self.batch_infer["infer_ts"])

            # sync agent
            if local_mac.agent_version < self.mac.agent_version:
                local_mac.load_agent_state_dict(self.mac.agent_state_dict)
                local_mac.agent_version = self.mac.agent_version.item()

            infer_data = self.episode_buffer.get_infer_data(infer_bs, infer_ts)
            with torch.inference_mode():
                result = local_mac.select_actions(infer_data, self.steps.item())
            self.episode_buffer.insert_data(result, infer_bs, infer_ts)
            self.steps += sum(self.batch_infer["alive_env"])

            terminated = self.episode_buffer[("terminated", infer_bs, infer_ts)]
            terminated_bs = np.where(terminated == 1)[0]
            if terminated_bs.size > 0:
                num_episode = terminated_bs.size
                episode_data = self.episode_buffer[infer_bs[terminated_bs]]
                shared_buffer_idx = self.queue_center.get_many("shared_idx", num_episode)
                self.shared_buffer.insert_episode(shared_buffer_idx, episode_data)
                self.queue_center.put("episode_idx", shared_buffer_idx)
            for idx, actor_id in enumerate(self.batch_infer["sample_id"]):
                if self.args.local:
                    self.queue_center.put_sample_request(actor_id)
                else:
                    target_slice = slice(idx * self.env_batch_size, (idx + 1) * self.env_batch_size)
                    self.queue_center.put("sample_request", [actor_id, infer_bs[target_slice], infer_ts[target_slice], self.steps.item()])
