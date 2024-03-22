import torch
from worker import worker_REGISTRY
from utils.queue import QueueCenter
from utils.buffer import SharedBuffer, EpisodeBuffer
from utils.scheme import scheme_REGISTRY
from utils.logger import Logger
from policy import policy_REGISTRY


def train_run(args):
    logger = Logger(args)

    last_log_T = 0
    last_evaluate_T = 0

    workers = []

    steps = torch.zeros(1, dtype=torch.int32)
    episodes = torch.zeros(1, dtype=torch.int32)

    buffer_scheme = scheme_REGISTRY["buffer"](args)
    transition_scheme = scheme_REGISTRY["transition"](args)

    episode_buffer = EpisodeBuffer(args, buffer_scheme)
    shared_buffer = SharedBuffer(args, buffer_scheme)
    queue_center = QueueCenter(args)

    mac = policy_REGISTRY[args.algorithm]["mac"]
    policy = policy_REGISTRY[args.algorithm]["policy"]
    global_mac = mac(args)
    global_mac.to(args.train_device)
    train_worker = worker_REGISTRY["train_worker"](args, global_mac, policy, queue_center, shared_buffer, steps, episodes)
    workers.append(train_worker)

    for _ in range(args.num_inference_worker):
        inference_worker = worker_REGISTRY["infer_worker"](args, global_mac, episode_buffer, queue_center, shared_buffer, steps)
        workers.append(inference_worker)

    if args.local:
        for i in range(args.num_sample_worker):
            sample_worker = worker_REGISTRY["local_sample_worker"](args, queue_center, episode_buffer, steps, worker_id=i)
            workers.append(sample_worker)
    else:
        infer_server = worker_REGISTRY["infer_server"](args, episode_buffer, queue_center, transition_scheme)
        workers.append(infer_server)

    if args.evaluate:
        evaluate_worker = worker_REGISTRY["evaluate_worker"](args, global_mac, queue_center, episode_buffer, steps)
        workers.append(evaluate_worker)

    for worker in workers:
        worker.start()

    while steps <= args.max_steps:
        log = queue_center.get("log")
        current_step = steps.item()
        current_episode = episodes.item()
        if log.get("type", None) == "info":
            logger.log_info(log["msg"])
        if log.get("type", None) == "data":
            logger.log_metrics(log["msg"], current_step)
        if (current_step - last_log_T) >= args.log_interval:
            logger.print_recent_stats(current_step, current_episode)
            last_log_T = current_step
        if current_step - last_evaluate_T >= args.evaluate_interval and args.evaluate:
            last_evaluate_T = current_step
            queue_center.put("evaluate_request", "evaluate")
    for worker in workers:
        worker.stop()
    logger.close()
