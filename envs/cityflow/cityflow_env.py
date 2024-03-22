import numpy as np
from smac.env import MultiAgentEnv
from cityflowenv.env import CityflowEnv as cfe

class CityFlowEnv(MultiAgentEnv):
    def __init__(self, **kwargs):
        self._env = cfe(**kwargs)
        self.n_agents = self._env.n_agents
        self.episode_limit = self._env.episode_limit

        self.reset()

    def reset(self):
        return self._env.reset()

    def step(self, action):
        return self._env.step(action)


    def get_state(self, near_state=False, with_att=False):
        return self._env.get_state()

    def get_current_time(self):
        return self._env.get_current_time()

    def get_avg_travel_time(self):
        """
        get average travel time
        """
        return self._env.get_avg_travel_time()

    def close(self):
        pass

    def get_state_size(self):
        """Returns the shape of the state"""
        return self._env.get_state_size()

    def get_obs_size(self):
        """Returns the shape of the observation"""
        return self._env.get_obs_size()

    def get_obs(self):
        return self._env.get_obs()

    def get_total_actions(self):
        return self._env.get_total_actions()

    def get_avail_actions(self):
        return self._env.get_avail_actions()

    def get_scheme(self):
        env_info = self.get_env_info()

        scheme = {
            "states": {"vshape": env_info["state_shape"], "dtype": np.float32},
            "obs": {
                "vshape": env_info["obs_shape"],
                "dtype": np.float32,
                "group": "agents",
            },
            "actions": {"vshape": (), "group": "agents", "dtype": np.int64},
            "available_actions": {
                "vshape": (env_info["n_actions"],),
                "group": "agents",
                "dtype": np.int32,
            },
            "rewards": {"vshape": (1,), "dtype": np.float32},
            "terminated": {"vshape": (1,), "dtype": np.int32},
        }
        groups = {"agents": env_info["n_agents"]}
        return scheme, groups
