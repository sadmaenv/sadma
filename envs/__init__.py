from functools import partial
from envs.smac import SMAC
from envs.replenishment import ReplenishmentEnv


def env_fn(env, **kwargs):
    return env(**kwargs)


REGISTRY = {}
REGISTRY["smac"] = partial(env_fn, env=SMAC)
REGISTRY["replenishment"] = partial(env_fn, env=ReplenishmentEnv)
