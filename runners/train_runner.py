from workers import worker_REGISTRY
from utils.queue_center import QueueCenter
from utils.buffers import SharedBuffer, EpisodeBuffer
from utils.scheme import scheme_REGISTRY
import torch
from controllers import REGISTRY as mac_REGISTRY
from utils.logger import Logger


def train_run(args):
    logger = Logger(args)

    last_log_T = 0
    last_evaluate_T = 0

    workers = []

    total_step = torch.zeros(1, dtype=torch.int32)
    total_episode = torch.zeros(1, dtype=torch.int32)

    buffer_scheme = scheme_REGISTRY["buffer"](args)
    transition_scheme = scheme_REGISTRY["transition"](args)

    shared_buffer = SharedBuffer(args, buffer_scheme)
    episode_buffer = EpisodeBuffer(args, buffer_scheme)
    queue_center = QueueCenter(args)

    global_mac = mac_REGISTRY[args.mac](args)
    global_mac.to(args.train_device)
    train_worker = worker_REGISTRY["train_worker"](args, global_mac, buffer_scheme, queue_center, shared_buffer)
    workers.append(train_worker)

    for _ in range(args.num_inference_worker):
        inference_worker = worker_REGISTRY["infer_worker"](args, global_mac, episode_buffer, queue_center, shared_buffer, total_step, total_episode)
        workers.append(inference_worker)

    if args.local:
        for i in range(args.num_sample_worker):
            sample_worker = worker_REGISTRY["local_sample_worker"](args, queue_center, episode_buffer, total_step, worker_id=i)
            workers.append(sample_worker)
    else:
        infer_server = worker_REGISTRY["infer_server"](args, episode_buffer, queue_center, transition_scheme)
        workers.append(infer_server)

    if args.evaluate:
        evaluate_worker = worker_REGISTRY["evaluate_worker"](args, global_mac, queue_center, episode_buffer, total_step)
        workers.append(evaluate_worker)

    for worker in workers:
        worker.start()

    while total_step <= args.t_max:
        log = queue_center.get("log")
        current_step = total_step.item()
        current_episode = total_episode.item()
        if log.get("type", None) == "info":
            logger.log_info(log["msg"])
        if log.get("type", None) == "data":
            logger.log_env_info(log["msg"], current_step)
        if (current_step - last_log_T) >= args.log_interval:
            logger.print_recent_stats(current_step, current_episode)
            last_log_T = current_step
        if current_step - last_evaluate_T >= args.evaluate_interval and args.evaluate:
            last_evaluate_T = current_step
            queue_center.put("evaluate_request", "evaluate")
    for worker in workers:
        worker.stop()
    logger.close()
