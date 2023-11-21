import torch
from utils.buffers import EpisodeBuffer, ReplayBuffer
from controllers import REGISTRY as mac_REGISTRY
from env_runners import runner_REGISTRY
from learners import learner_REGISTRY
from utils.scheme import scheme_REGISTRY
import numpy as np


def sync_run(args):
    buffer_scheme = scheme_REGISTRY["buffer"](args)
    mac = mac_REGISTRY[args.mac](args)
    mac.to(args.train_device)
    learner = learner_REGISTRY[args.learner](args, mac)
    episode_buffer = EpisodeBuffer(args, buffer_scheme)
    replay_buffer = ReplayBuffer(args, buffer_scheme)
    env_runner = runner_REGISTRY[args.env_type](args)
    episode_count = 0
    total_episode = 0

    cpu_actions = None

    bs = list(range(0, args.env_batch_size))
    while env_runner.env_total_step < args.t_max:
        step_data, _ = env_runner.step(cpu_actions)
        episode_buffer.insert_data(step_data, bs=bs, ts=env_runner.envs_step)
        infer_data = episode_buffer.get_infer_data(bs, env_runner.envs_step)
        with torch.no_grad():
            result = mac.select_actions(infer_data, env_runner.env_total_step)
        cpu_actions = result["actions"].cpu().numpy()
        episode_buffer.insert_data(result, bs, env_runner.envs_step)
        terminated = episode_buffer[("terminated", bs, env_runner.envs_step)]
        terminated_bs = np.where(terminated == 1)[0]
        if terminated_bs.size > 0:
            episode_data = episode_buffer[terminated_bs]
            num_episode = terminated_bs.size
            replay_buffer.insert_episodes(num_episode, episode_data)
            episode_count += num_episode
            total_episode += num_episode
            if episode_count >= args.update_rate and replay_buffer.can_sample():
                episode_count = 0
                train_data = replay_buffer.sample()
                learner.train(train_data, total_episode)
                del train_data
        if np.all(env_runner.terminated):
            print("reset")
            episode_buffer.reset_data(bs)
    env_runner.close_env()
