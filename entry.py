import torch
import numpy as np
from runner import runner_REGISTRY
from config.config_utils import get_all_config

if __name__ == '__main__':
    torch.multiprocessing.set_start_method("spawn")
    torch.set_num_threads(1)
    args = get_all_config()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.env_args['seed'] = args.seed

    if args.async_train:
        if args.role == "train":
            runner_REGISTRY["train"](args)
        elif args.role == "sample":
            runner_REGISTRY["sample"](args)
    else:
        runner_REGISTRY["sync"](args)
