import json
import logging
import os
from torch.utils.tensorboard import SummaryWriter
import copy
import numpy as np
from collections import defaultdict
import torch


class Logger:
    def __init__(self, args):
        self.args = args
        self.info_logger = None
        self.tensor_logger = None
        self.setup(args)
        self.stats = defaultdict(lambda: [])

    def setup(self, args):
        # logger
        self.info_logger = logging.getLogger("logger")
        handlers = []
        formatter = logging.Formatter('[%(levelname)s %(asctime)s] %(message)s', '%H:%M:%S')
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        handlers.append(sh)

        if args.save_log:
            # mkdir
            path_root = os.path.join(os.getcwd(), args.log_root_path, args.algorithm, args.map_name, args.unique_token)
            mkdirs = lambda folder_path: os.makedirs(folder_path, exist_ok=True) if not os.path.exists(folder_path) else None
            mkdirs(path_root)
            # save args
            args_log = copy.deepcopy(args)
            get_numpy_type_name = lambda x: np.zeros(1, dtype=x).dtype.name
            for k, v in args_log.scheme.items():
                args_log.scheme[k]["dtype"] = get_numpy_type_name(v["dtype"])
            args_json = json.dumps(vars(args_log), indent=4)
            with open(os.path.join(path_root, "args.json"), "w") as json_file:
                json_file.write(args_json)
            # add FileHandler
            fh_path = os.path.join(path_root, "log.txt")
            fh = logging.FileHandler(filename=fh_path)
            fh.setFormatter(formatter)
            handlers.append(fh)
            # add tensorboard
            if args.use_tensorboard:
                self.tensor_logger = SummaryWriter(os.path.join(path_root, "tensorboard"))

        for handler in handlers:
            self.info_logger.addHandler(handler)
        self.info_logger.setLevel(logging.INFO)

    def log_env_info(self, env_infos, total_step):
        for env_info in env_infos:
            for k, v in env_info.items():
                self.stats[k].append((total_step, v))

    def log_info(self, msg):
        self.info_logger.info(msg)
    
    def log_tensorboard(self, step):
        for (k, v) in sorted(self.stats.items()):
            item = torch.mean(torch.tensor([float(x[1]) for x in self.stats[k]])).item()
            self.tensor_logger.add_scalar(k, item, step)
            
    def print_recent_stats(self, step, episode):
        log_str = "Recent Stats | t_env: {:>10} | Episode: {:>8}\n".format(step, episode)
        i = 0
        for (k, v) in sorted(self.stats.items()):
            if k == "episode":
                continue
            i += 1
            window = 5 if k != "epsilon" else 1
            item = "{:.4f}".format(torch.mean(torch.tensor([float(x[1]) for x in self.stats[k][-window:]])))
            log_str += "{:<25}{:>8}".format(k + ":", item)
            log_str += "\n" if i % 4 == 0 else "\t"
        self.log_info(log_str)
        if self.args.use_tensorboard:
            self.log_tensorboard(step)
        # Reset stats to avoid accumulating logs in memory
        self.stats = defaultdict(lambda: [])

    def close(self):
        for handler in self.info_logger.handlers:
            handler.close()
