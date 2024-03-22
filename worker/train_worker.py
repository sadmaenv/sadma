import torch
from torch.multiprocessing import Process
from utils.buffer import ReplayBuffer
from worker.base_worker import ProcessWorker


class TrainWorker(ProcessWorker):
    def __init__(self, args, global_mac, policy, queue_center, shared_buffer, steps, episodes):
        super().__init__()
        self.args = args
        self.global_mac = global_mac
        self.queue_center = queue_center
        self.shared_buffer = shared_buffer
        self.replay_buffer = ReplayBuffer(args, shared_buffer.scheme)
        self.policy = policy(args, global_mac)
        self.episodes = episodes
        self.steps = steps
        self.job = Process(target=self.async_train)


    def async_train(self):
        torch.set_num_threads(1)
        self.queue_center.put("log", {"type": "info", "msg": "train worker start"})
        last_update_episode = 0
        while True:
            episode_idx = self.queue_center.get("episode_idx")
            batch_size = len(episode_idx)
            episode_data = self.shared_buffer[episode_idx]
            self.replay_buffer.insert_episodes(batch_size, episode_data)
            self.queue_center.put_many("shared_idx", episode_idx)
            self.episodes += batch_size
            if self.episodes.item() - last_update_episode >= self.args.update_rate and self.replay_buffer.can_sample():
                last_update_episode = self.episodes.item()
                train_data = self.replay_buffer.sample()
                metrics = self.policy.train(train_data, self.episodes.item(), self.steps.item())
                self.queue_center.put("log", {"type": "data", "msg": [metrics]})
