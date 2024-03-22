import os
import time
import yaml
import argparse
import collections
import copy
from types import SimpleNamespace as SN
from utils.utils import mkdir
from envs import env_REGISTRY

def get_config_yaml(path):
    with open(path, "r") as f:
        try:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)
    return config_dict

def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return copy.deepcopy(config)

def update_args(args):
    # 将时间当前戳转换为自定义格式的字符串
    formatted_time = time.strftime("%Y%m%d-%H:%M:%S", time.localtime(time.time()))
    args.unique_token = f"seed{args.seed}_{formatted_time}"
    args.log_path = os.path.join(os.getcwd(), args.log_root_path, args.algorithm, args.map_name, args.unique_token)
    if args.save_log:
        mkdir(args.log_path)
    if args.env == "cityflow":
        mkdir(args.log_path)
        args.env_args["log_path"] = args.log_path  # for cityflow to save replay file
    # add env info
    env = env_REGISTRY[args.env](**args.env_args)
    env_info = env.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.obs_shape = env_info["obs_shape"]
    args.episode_limit = env_info["episode_limit"]
    args.episode_length = env_info["episode_limit"] + 1
    scheme, groups = env.get_scheme()
    args.scheme = scheme
    args.groups = groups
    env.close()
    # add address info
    if args.local:
        args.address = "127.0.0.1"
    else:
        if args.role == "train":
            args.address = os.environ.get("PodIp")
        if args.role == "sample":
            args.address = os.environ.get("LeaderAddress")
    return args

def get_input_args(configs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="", help="experiment name")
    parser.add_argument("--seed", type=int, default=1, help="seed of the experiment")
    parser.add_argument("--alg", type=str, default="ppo", help="algorithm")
    parser.add_argument("--env", type=str, default="cityflow", help="env type: smac cityflow powergrid")
    parser.add_argument("--map_name", type=str, default="jinan", help="map name. 3m cityflow:[jinan, hangzhou, newyork]")
    # parser.add_argument("--map-name", type=str, default="sku1000.single_store.standard", help="map name")
    parser.add_argument("--train_device", type=str, default="cuda:0", help="train device")
    parser.add_argument("--env_batch_size", type=int, default=8, help="number of parallel env batch size")

    # distribute
    parser.add_argument("--async_train", type=bool, default=False, help="if async train")
    parser.add_argument("--local", type=bool, default=True, help="local or remote")
    parser.add_argument("--role", type=str, default="train", help="sample or train, when use remote",)
    parser.add_argument("--num_sample_worker", type=int, default=1, help="number of sample worker")
    parser.add_argument("--sampler_id", type=int, default=0, help="when use remote, sampler id for assign port",)
    args = parser.parse_args()
    
    configs["exp_name"] = args.exp_name
    configs["seed"] = args.seed
    configs["algorithm"] = args.alg
    configs["env"] = args.env
    configs["map_name"] = args.map_name
    configs["train_device"] = args.train_device
    configs["env_batch_size"] = args.env_batch_size

    configs["role"] = args.role
    configs["async_train"] = args.async_train
    configs["num_sample_worker"] = args.num_sample_worker
    configs["local"] = args.local

    return configs

def get_all_config():
    config_root_path = os.path.dirname(__file__)
    # load default args
    default_yaml_path = os.path.join(config_root_path, "default.yaml")
    configs = get_config_yaml(default_yaml_path)
    configs = get_input_args(configs)
    # load algorithm args
    alg = configs["algorithm"]
    alg_yaml_path = os.path.join(config_root_path, "alg_config", f"{alg}.yaml")
    alg_config = get_config_yaml(alg_yaml_path)
    configs = recursive_dict_update(configs, alg_config)
    # load env args
    env = configs["env"]
    env_yaml_path = os.path.join(config_root_path, "env_config", f"{env}.yaml")
    env_config = get_config_yaml(env_yaml_path)
    if env == "replenishment":
        env_config["env_args"]["task_type"] = configs["map_name"]
    env_config["env_args"]["map_name"] = configs["map_name"]
    env_config["env_args"]["seed"] = configs["seed"]
    configs = recursive_dict_update(configs, env_config)

    args = SN(**configs)
    args = update_args(args)

    return args