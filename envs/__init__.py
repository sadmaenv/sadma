from functools import partial
from envs.smac.smac_env import SMAC
from envs.replenishment.replenishment_env import ReplenishmentEnv
from envs.cityflow.cityflow_env import CityFlowEnv
from envs.powergrid.smart_grid import GridEnv

def env_fn(env, **kwargs):
    return env(**kwargs)

env_REGISTRY = {}
env_REGISTRY["smac"] = partial(env_fn, env=SMAC)
env_REGISTRY["replenishment"] = partial(env_fn, env=ReplenishmentEnv)
env_REGISTRY["cityflow"] = partial(env_fn, env=CityFlowEnv)
env_REGISTRY["powergrid"] = partial(env_fn, env=GridEnv)

from envs.smac.smac_runner import SMACRunner
from envs.replenishment.replenishment_runner import ReplenishmentRunner
from envs.cityflow.cityflow_runner import CityFlowRunner
from envs.powergrid.powergrid_runner import PowerGridRunner
env_runner_REGISTRY = {}
env_runner_REGISTRY["smac"] = SMACRunner
env_runner_REGISTRY["replenishment"] = ReplenishmentRunner
env_runner_REGISTRY["cityflow"] = CityFlowRunner
env_runner_REGISTRY["powergrid"] = PowerGridRunner