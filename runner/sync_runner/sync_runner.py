import numpy as np
import torch
from utils.buffer import EpisodeBuffer, ReplayBuffer
from policy import policy_REGISTRY
from envs import env_runner_REGISTRY
from utils.scheme import scheme_REGISTRY
from utils.logger import Logger


def sync_run(args):
    logger = Logger(args)
    buffer_scheme = scheme_REGISTRY["buffer"](args)
    mac = policy_REGISTRY[args.algorithm]["mac"](args)
    mac.to(args.train_device)
    policy = policy_REGISTRY[args.algorithm]["policy"](args, mac)
    episode_buffer = EpisodeBuffer(args, buffer_scheme)
    replay_buffer = ReplayBuffer(args, buffer_scheme)
    env_runner = env_runner_REGISTRY[args.env](args)
    last_update_episode = 0
    last_log_T = 0
    episodes = 0
    steps = 0

    actions = None
    bs = np.array(range(0, args.env_batch_size))
    while steps < args.max_steps:
        step_data, env_metrics = env_runner.step(actions)
        steps += args.env_batch_size
        logger.log_metrics(env_metrics, env_runner.envs_step)
        episode_buffer.insert_data(step_data, bs=bs, ts=env_runner.envs_step)
        infer_data = episode_buffer.get_infer_data(bs, env_runner.envs_step)
        with torch.no_grad():
            result = mac.select_actions(infer_data, steps)
        actions = result["actions"].numpy()
        episode_buffer.insert_data(result, bs, env_runner.envs_step)
        terminated = episode_buffer[("terminated", bs, env_runner.envs_step)]
        terminated_bs = np.where(terminated == 1)[0]
        if terminated_bs.size > 0:
            episode_data = episode_buffer[terminated_bs]
            num_episode = terminated_bs.size
            replay_buffer.insert_episodes(num_episode, episode_data)
            episodes += num_episode
            if episodes - last_update_episode >= args.update_rate and replay_buffer.can_sample():
                last_update_episode = episodes
                train_data = replay_buffer.sample()
                train_metrics = policy.train(train_data, episodes, steps)
                logger.log_metrics(train_metrics, steps)
        if np.all(env_runner.terminated):
            episode_buffer.reset_data(bs)
        if (episodes - last_log_T) >= args.log_interval_episode:
            last_log_T = episodes
            logger.print_recent_stats(steps, episodes)
    env_runner.close_env()
