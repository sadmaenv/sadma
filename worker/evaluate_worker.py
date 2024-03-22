import copy
import torch
import numpy as np
from torch.multiprocessing import Process
from envs import env_runner_REGISTRY
from worker.base_worker import ProcessWorker


class EvaluateWorker(ProcessWorker):
    def __init__(self, args, global_mac, queue_center, episode_buffer, steps):
        super(EvaluateWorker, self).__init__()
        self.args = args
        self.global_mac = global_mac
        self.queue_center = queue_center
        self.episode_buffer = episode_buffer
        self.device = args.train_device
        self.steps = steps
        self.job = Process(target=self.evaluate)

    def evaluate(self):
        torch.set_num_threads(1)
        local_mac = copy.deepcopy(self.global_mac)
        local_mac.to(self.device)
        self.queue_center.put("log", {"type": "info", "msg": "evaluate worker start"})
        env_runner = env_runner_REGISTRY[self.args.env_type](self.args, evaluate=True)
        bs = list(range(self.episode_buffer.buffer_size))[-self.args.evaluate_batch_size :]
        while True:
            _ = self.queue_center.get("evaluate_request")
            env_runner.reset_all_env()
            self.episode_buffer.reset_data(bs)
            terminated_env_infos = []
            if local_mac.agent_version < self.global_mac.agent_version:
                local_mac.load_agent_state(self.global_mac.agent_state())
                local_mac.agent_version = self.global_mac.agent_version.item()

            cpu_actions = None
            while True:
                env_return, terminated_env_info = env_runner.step(cpu_actions)
                self.episode_buffer.insert_data(env_return, bs=bs, ts=env_runner.envs_step)
                terminated_env_infos += terminated_env_info
                if len(terminated_env_infos) >= self.args.evaluate_episodes:
                    terminated_env_infos = terminated_env_infos[:self.args.evaluate_episodes]
                    break
                infer_data = self.episode_buffer.get_infer_data(bs, env_runner.envs_step)
                with torch.no_grad():
                    result = local_mac.select_action(infer_data, self.total_step, test_mode=True)
                cpu_actions = result["action"].cpu().numpy()
                self.episode_buffer.insert_data(result, bs, env_runner.envs_step)
                if np.all(env_runner.terminated):
                    self.episode_buffer.reset_data(bs)

            log_env_info = {"type": "data", "msg": terminated_env_infos}
            self.queue_center.put("log", log_env_info)