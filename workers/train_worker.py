from learners import learner_REGISTRY
from utils.buffers import ReplayBuffer
from torch.multiprocessing import Process
from workers.base_worker import BaseWorker
import torch


class TrainWorker(BaseWorker):
    def __init__(self, args, global_mac, buffer_scheme, queue_center, shared_buffer):
        super().__init__()
        self.args = args
        self.global_mac = global_mac
        self.buffer_scheme = buffer_scheme
        self.queue_center = queue_center
        self.shared_buffer = shared_buffer
        self.replay_buffer = ReplayBuffer(args, buffer_scheme)
        self.learner = learner_REGISTRY[args.learner](args, global_mac, queue_center)
        self.total_episode = 0
        self.cur_episode = 0
        self.job = Process(target=self.async_train)

    def async_train(self):
        torch.set_num_threads(1)
        self.queue_center.put("log", {"type": "info", "msg": "train worker start"})
        while True:
            episode_idx = self.queue_center.get("episode_idx")
            batch_size = len(episode_idx)
            episode_data = self.shared_buffer[episode_idx]
            self.replay_buffer.insert_episodes(batch_size, episode_data)
            self.queue_center.put_many("shared_idx", episode_idx)
            self.total_episode += batch_size
            self.cur_episode += batch_size
            if self.cur_episode >= self.args.update_rate and self.replay_buffer.can_sample():
                self.cur_episode = 0
                train_data = self.replay_buffer.sample()
                self.learner.train(train_data, self.total_episode)
                del train_data
