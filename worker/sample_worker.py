import torch
import numpy as np
from torch.multiprocessing import Process

from envs import env_runner_REGISTRY
from utils.scheme import scheme_REGISTRY
from worker.base_worker import ProcessWorker
from worker.zmq_worker import InferClient


class LocalSampleWorker(ProcessWorker):
    def __init__(self, args, queue_center, episode_buffer, steps, worker_id=0):
        super().__init__()
        self.args = args
        self.queue_center = queue_center
        self.episode_buffer = episode_buffer
        self.steps = steps
        self.worker_id = worker_id
        self.env_batch_size = args.env_batch_size

        self.job = Process(target=self.without_model_sample)

    def without_model_sample(self):
        torch.set_num_threads(1)
        self.queue_center.put("log", {"type": "info", "msg": f"sample{self.worker_id} worker start"})
        stop_sample = False
        if self.args.num_env_runner == 1:
            env_runner = env_runner_REGISTRY[self.args.env](self.args, evaluate=False)
            actions = None
            bs = np.arange(self.env_batch_size * self.worker_id, self.env_batch_size * (self.worker_id + 1))
            while not stop_sample:
                env_return, terminated_env_info = env_runner.step(actions)
                if np.all(env_runner.envs_step == 0):
                    self.episode_buffer.reset_data(bs)
                self.episode_buffer.insert_data(env_return, bs=bs, ts=env_runner.envs_step)
                infer_request = {
                    "infer_bs": bs,
                    "infer_ts": env_runner.envs_step,
                    "sample_id": self.worker_id,
                    "alive_env": env_runner.alive_env(),
                }
                self.queue_center.put("infer_request", infer_request)
                if terminated_env_info:
                    self.queue_center.put("log", {"type": "data", "msg": terminated_env_info})
                _ = self.queue_center.get_sample_request(self.worker_id)
                actions = self.episode_buffer[("actions", bs, env_runner.envs_step)]
                actions = actions.squeeze().cpu().numpy()
            env_runner.close_env()


class RemoteSampleWorker(ProcessWorker):
    def __init__(self, args, worker_id=0):
        super().__init__()
        self.args = args
        self.worker_id = worker_id
        self.job = Process(target=self.without_model_sample)

    def without_model_sample(self):
        torch.set_num_threads(1)
        data_port = self.args.base_port + self.worker_id
        transition_scheme = scheme_REGISTRY["transition"](self.args)
        client = InferClient(data_port, transition_scheme, self.args.address)
        stop_sample = False
        if self.args.num_env_runner == 1:
            env_runner = env_runner_REGISTRY[self.args.env](self.args, evaluate=False)
            actions = None
            while not stop_sample:
                env_return, terminated_env_info = env_runner.step(actions)
                envs_step = env_runner.envs_step
                alive_env = env_runner.alive_env()
                actions, total_steps = client.send_recv(env_return, envs_step, alive_env, terminated_env_info)
                if isinstance(actions, str):
                    stop_sample = True
            env_runner.close_env()
