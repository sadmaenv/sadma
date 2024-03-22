import copy
import json
import logging
import os
from collections import defaultdict
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from utils.utils import mkdir


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
        formatter = logging.Formatter("[%(levelname)s %(asctime)s] %(message)s", "%H:%M:%S")
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        handlers.append(sh)

        if args.save_log:
            # mkdir
            path_root = os.path.join(
                os.getcwd(),
                args.log_root_path,
                args.algorithm,
                args.map_name,
                args.unique_token,
            )
            mkdir(path_root)
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

    def log_metrics(self, metrics, total_steps):
        if isinstance(metrics, list) and len(metrics) > 0:
            for metric in metrics:
                for k, v in metric.items():
                    self.stats[k].append((total_steps, v))
        elif isinstance(metrics, dict):
            for k, v in metrics.items():
                self.stats[k].append((total_steps, v))

    def log_info(self, msg):
        self.info_logger.info(msg)

    def log_tensorboard(self, step):
        for (k, v) in sorted(self.stats.items()):
            item = torch.mean(torch.tensor([float(x[1]) for x in self.stats[k]])).item()
            self.tensor_logger.add_scalar(k, item, step)

    def print_recent_stats(self, step, episode):
        log_str = "Recent Stats | t_env: {:>10} | Episode: {:>8}\n".format(step, episode)
        i = 0
        for k, v in sorted(self.stats.items()):
            if k == "episode":
                continue
            i += 1
            item = "{:.4f}".format(torch.mean(torch.tensor([float(x[1]) for x in self.stats[k]])))
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
