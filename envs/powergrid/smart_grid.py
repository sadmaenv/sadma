import json
import os
import pickle
import warnings

import numpy as np
import pandapower as pp
import pandapower.networks as networks
import yaml
from numpy.linalg import matrix_power
from pandapower import runpp

from smac.env import MultiAgentEnv

from .energy_models import Building, Weather
import time

warnings.filterwarnings("ignore")


class GridEnv(MultiAgentEnv):
    def __init__(
        self,
        data_path="envs/powergrid/data/Climate_Zone_1",
        climate_zone="Climate_Zone_1",
        buildings_states_actions_file="envs/powergrid/data/buildings_state_action_space.json",
        hourly_timesteps=4,
        houses_per_node=6,
        cluster_adjacent_bus_num=6,
        save_memory=True,
        building_ids=None,
        nclusters=2,
        randomseed=2,
        max_num_houses=None,
        percent_rl=1,
        net_path="envs/powergrid/data/640_agents/case33.p",
        agent_path="envs/powergrid/data/640_agents/agent_640_zone_1.pickle",
        **kwargs,
    ):
        self.max_num_houses = max_num_houses
        #        self.nclusters = nclusters
        self.percent_rl = percent_rl

        self.cluster_adjacent_bus_num = cluster_adjacent_bus_num

        self.data_path = data_path
        self.climate_zone = climate_zone
        self.weather_file = os.path.join(self.data_path, "weather_data.csv")
        self.solar_file = os.path.join(self.data_path, "solar_generation_1kW.csv")
        self.weather = Weather(self.weather_file, self.solar_file, hourly_timesteps)
        self.buildings_states_actions_file = buildings_states_actions_file
        self.hourly_timesteps = hourly_timesteps
        self.save_memory = save_memory
        self.building_ids = building_ids
        self.exipode = 0
        bt = time.time()
        self.net = self._make_grid()
        # self.net = pp.from_pickle(net_path)
        self.buildings = self._add_houses(houses_per_node, 1)  # standard 6 buildings
        # with open(agent_path, "rb") as f:
        #     self.buildings = pickle.load(f)
        self.agents = list(self.buildings.keys())
        self.possible_agents = self.agents[:]
        self.rl_agents = self._set_rl_agents()

        self.clusters = self._get_bus_clusters()

        self.observation_spaces = {
            k: v.observation_space for k, v in self.buildings.items()
        }
        self.action_spaces = {k: v.action_space for k, v in self.buildings.items()}

        self.metadata = {"render.modes": [], "name": "gridlearn"}

        self.voltage_data = []
        self.load_data = []
        self.gen_data = []
        self.reward_data = []
        self.vm_reward_data = []
        self.all_rewards = []

        self.aspace, self.ospace = self._get_spaces(self.agents)
        self.single_agent_obs_size = self.ospace[self.agents[0]].shape[0]
        self.obs_size = self._get_partial_obs_max_len()
        self.state_size = self.single_agent_obs_size * len(self.agents)

        self.v_upper = 1.05
        self.v_lower = 0.95

        self.n_agents = len(self.agents)
        # with open("./config.yaml") as file:
        #     config = yaml.safe_load(file)
        # self.episode_limit = config["environment"]["max_cycles"]  # 5 days
        self.episode_limit = 92  # 92

    def get_env_info(self):
        env_info = {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "n_actions": self.get_total_actions(),
            "action_space": self.buildings[self.agents[0]].action_space,
            "agents_name": self.agents,
            "n_agents": self.n_agents,
            "episode_limit": self.episode_limit,
            "action_shape": (4,),
        }
        return env_info

    def set_reward_net(self, reward_net):
        self.reward_net = reward_net

    def reset(self, reset_logs=True):
        # self.system_losses = []
        # self.voltage_dev = []
        # return {k:self.buildings[k].reset_timestep(self.net, reset_logs) for k in agents}
        rand_act = {k: v.sample() for k, v in self.aspace.items()}
        rand_act_array = np.array(list(rand_act.values()))
        self.steps = 0
        self.ts = self._select_timestamp(self.exipode)
        self.exipode += 100
        # print("self.ts", self.ts)
        year_ts = self.ts % (8760 * self.hourly_timesteps)
        if year_ts > 90 * 96 and year_ts < 275 * 96:
            self.net.shunt.at[0, "q_mvar"] = -1.8
            self.net.shunt.at[1, "q_mvar"] = -0.6
            self.net.shunt.at[2, "q_mvar"] = -1.2
        else:
            self.net.shunt.at[0, "q_mvar"] = -1.2
            self.net.shunt.at[1, "q_mvar"] = -0.01
            self.net.shunt.at[2, "q_mvar"] = -0.01

        count = 0
        for agent in self.agents:
            self.buildings[agent].step(rand_act_array[count])
            count += 1

        self.ts += 1
        self.steps += 1

        # update the grid based on updated buildings
        self._update_grid()

        # run the grid power flow
        try:
            runpp(self.net, enforce_q_lims=True)
            # print(self.net.res_bus.p_mw.tolist())
        except:
            pp.diagnostic(self.net)
            quit()
            print("QUITTING!!!!")

        for agent in self.agents:
            self.buildings[agent].reset_timestep(self.net, reset_logs)

        return self.get_obs(), self.get_state()

    def step(self, actions):
        year_ts = self.ts % (8760 * self.hourly_timesteps)
        if year_ts > 90 * 96 and year_ts < 275 * 96:
            self.net.shunt.at[0, "q_mvar"] = -1.8
            self.net.shunt.at[1, "q_mvar"] = -0.6
            self.net.shunt.at[2, "q_mvar"] = -1.2
        else:
            self.net.shunt.at[0, "q_mvar"] = -1.2
            self.net.shunt.at[1, "q_mvar"] = -0.01
            self.net.shunt.at[2, "q_mvar"] = -0.01

        count = 0

        for agent in self.agents:
            self.buildings[agent].step(actions[count])
            count += 1

        self.ts += 1
        self.steps += 1

        # update the grid based on updated buildings
        self._update_grid()

        # run the grid power flow
        try:
            runpp(self.net, enforce_q_lims=True)
            # print(self.net.res_bus.p_mw.tolist())
        except:
            pp.diagnostic(self.net)
            quit()
            print("QUITTING!!!!")
        rl_agent_keys = self.agents
        #        obs = self.state(rl_agent_keys)
        self.voltage_data += [list(self.net.res_bus["vm_pu"])]

        self.load_data += [sum(list(self.net.load["p_mw"]))]
        self.gen_data += [sum(list(self.net.sgen["p_mw"]))]
        # reward返回的是真实reward, vm_reward返回的是逆强化学习得到的奖励方程的估计值
        rewards, vm_rewards = self._get_reward()
        return np.mean(rewards), self._get_done(), self._get_info()  # , vm_rewards

    def get_rewardnet(self, action):
        observation_before = self.get_obs()
        observation_before = np.array(observation_before)
        net_reward = self.reward_net.compute_reward(
            np.concatenate([observation_before, np.clip(action, -1.0, 1.0)], axis=1)
        )
        # print('@@@@@@@@@@@@@@@@@@@', np.shape(np.concatenate([observation_before, np.clip(action, -1., 1.)], axis=1)))
        reward = net_reward  # - ctrl_cost
        return reward

    def get_state(self):
        state = np.concatenate(
            [np.array(self.buildings[k].get_state(self.net)) for k in self.agents]
        )
        return state

    def get_obs(self):
        all_state_dict = {
            k: np.array(self.buildings[k].get_state(self.net)) for k in self.agents
        }
        pad_obs_list = []
        for agent in self.rl_agents:
            agent_obs_array = np.concatenate(
                [all_state_dict[neighbor] for neighbor in self.clusters[agent]]
            )
            # print(
            #     "#############neighbor:", agent, self.obs_size, agent_obs_array.shape[0]
            # )
            pad_obs_list.append(
                np.concatenate(
                    [
                        agent_obs_array,
                        np.zeros(self.obs_size - agent_obs_array.shape[0]),
                    ]
                )
            )
        return np.array(pad_obs_list)

    def get_obs_agent(self, agent_id):
        """return observation for agent_id"""
        agents_obs = self.get_obs()
        return agents_obs[agent_id]

    def get_obs_size(self):
        """return the observation size"""
        return self.obs_size

    def get_state_size(self):
        """return the state size"""
        return self.state_size

    def get_num_of_agents(self):
        return len(self.agents)

    def get_total_actions(self):
        return self.aspace[self.agents[0]].shape[0]

    def get_avail_actions(self):
        avail_actions = []
        for _ in range(len(self.agents)):
            avail_actions.append([1] * self.get_total_actions())
        return avail_actions

    def _select_timestamp(self, exipode):
        data_len = len(self.weather.data["t_out"])
        time_stamp = np.random.choice(data_len - self.episode_limit)
        time_stamp = 5087
        time_stamp = 192 + exipode
        return time_stamp

    def _get_spaces(self, agents):
        actionspace = {k: self.buildings[k].action_space for k in agents}
        obsspace = {k: self.buildings[k].observation_space for k in agents}
        return actionspace, obsspace

    def _set_rl_agents(self):
        num_rl_agents = int(self.percent_rl * len(self.net.load.name))
        rl_agents = np.random.choice(self.net.load.name, num_rl_agents).tolist()
        return rl_agents

    def _make_grid(self):
        # make a grid that fits the buildings generated for CityLearn
        net = networks.case33bw()

        # clear the grid of old load values
        load_nodes = net.load["bus"]
        res_voltage_nodes = net.bus["name"][net.bus["vn_kv"] == 12.66]
        res_load_nodes = set(load_nodes) & set(res_voltage_nodes)
        net.bus["min_vm_pu"] = 0.7
        net.bus["max_vm_pu"] = 1.3

        for node in res_load_nodes:
            # remove the existing arbitrary load
            net.load.drop(net.load[net.load.bus == node].index, inplace=True)

        conns = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
            [18, 19, 20, 21],
            [25, 26, 27, 28, 29, 30, 31, 32],
            [22, 23, 24],
        ]

        self.pv_buses = [item[-1] for item in conns]
        self.pv_buses += [item[-2] for item in conns]

        mapping = {18: 1, 25: 5, 22: 2}

        net.line.drop(index=net.line[net.line.in_service == False].index, inplace=True)
        net.bus_geodata.at[0, "x"] = 0
        net.bus_geodata.at[0, "y"] = 0
        sw = "x"
        st = "y"
        z = -1
        for c in conns:
            z += 1
            for i in range(len(c)):
                if i == 0:
                    if not c[i] == 0:
                        sw = "y"
                        st = "x"
                        net.bus_geodata.at[c[i], sw] = (
                            net.bus_geodata.at[mapping[c[i]], sw] + 0.2
                        )
                        net.bus_geodata.at[c[i], st] = net.bus_geodata.at[
                            mapping[c[i]], st
                        ]
                else:
                    net.bus_geodata.at[c[i], sw] = (
                        net.bus_geodata.at[c[i - 1], sw] + 0.2
                    )
                    net.bus_geodata.at[c[i], st] = net.bus_geodata.at[c[i - 1], st]

        net.ext_grid.at[0, "vm_pu"] = 1.01

        pp.create_shunt_as_capacitor(net, 14, 1.2, 0)
        pp.create_shunt_as_capacitor(net, 24, 0.6, 0)
        pp.create_shunt_as_capacitor(net, 30, 1.2, 0)
        return net

    def _add_houses(self, n, pv_penetration):
        if self.max_num_houses:
            m = 1
        else:
            m = n  # + np.random.randint(-2,8)
        houses = []
        b = 0
        scaling_number = 6 / n

        # find nodes in the network with residential voltage levels and load infrastructure
        # get the node indexes by their assigned names
        # load_nodes = self.net.load['bus']
        ext_grid_nodes = set(self.net.ext_grid["bus"])
        res_voltage_nodes = set(self.net.bus["name"][self.net.bus["vn_kv"] == 12.66])
        res_load_nodes = res_voltage_nodes - ext_grid_nodes

        buildings = {}
        for existing_node in list(res_load_nodes)[: self.max_num_houses]:
            # remove the existing arbitrary load
            self.net.load.drop(
                self.net.load[self.net.load.bus == existing_node].index, inplace=True
            )

            # add n houses at each of these nodes
            BuildingId = 0
            for i in range(m):
                # bid = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5))
                BuildingId += 1
                if not self.building_ids:
                    with open(self.buildings_states_actions_file) as file:
                        buildings_states_actions = json.load(file)
                    self.building_ids = list(buildings_states_actions.keys())
                    # print(self.building_ids)
                prob = np.ones(len(self.building_ids))
                prob[
                    [1, 4, 5, 6, 7, 8]
                ] = 10  # building at index 0, 2, 8 correspond to buildings with high energy use
                prob = prob / sum(prob)
                uid = np.random.choice(self.building_ids, p=prob)
                # print(uid)
                bldg = Building(
                    self.data_path,
                    self.climate_zone,
                    self.buildings_states_actions_file,
                    self.hourly_timesteps,
                    uid,
                    self.weather,
                    BuildingId,
                    save_memory=self.save_memory,
                )
                bldg.assign_bus(existing_node)
                bldg.load_index = pp.create_load(
                    self.net, bldg.bus, 0, name=bldg.buildingId, scaling=scaling_number
                )  # create a load at the existing bus
                if np.random.uniform() <= 2:  # equivalent to 100% PV penetration
                    bldg.gen_index = pp.create_sgen(
                        self.net,
                        bldg.bus,
                        0,
                        name=bldg.buildingId,
                        scaling=scaling_number,
                    )  # create a generator at the existing bus
                else:
                    bldg.gen_index = -1

                buildings[bldg.buildingId] = bldg
                # bldg.assign_neighbors(self.net)

        from collections import Counter

        types = [v.building_type for v in buildings.values()]
        # print(Counter(types))

        return buildings

    def _update_grid(self):
        for agent, bldg in self.buildings.items():
            # Assign the load in MW (from KW in CityLearn)
            self.net.load.at[bldg.load_index, "p_mw"] = (
                0.95 * bldg.current_gross_electricity_demand * 0.001
            )
            self.net.load.at[bldg.load_index, "sn_mva"] = (
                bldg.current_gross_electricity_demand * 0.001
            )

            if (
                bldg.gen_index > -1
            ):  # assume PV and battery are both behind the inverter
                self.net.sgen.at[bldg.gen_index, "p_mw"] = (
                    -1 * bldg.current_gross_generation * np.cos(bldg.phi) * 0.001
                )
                self.net.sgen.at[bldg.gen_index, "q_mvar"] = (
                    bldg.current_gross_generation * np.sin(bldg.phi) * 0.001
                )

    def _get_reward(self):
        rewards = {k: self.buildings[k].get_reward(self.net) for k in self.agents}
        vm_rewards = {k: self.buildings[k].get_doreward(self.net) for k in self.agents}
        self.reward_data += [sum(rewards.values())]
        self.vm_reward_data += [sum(vm_rewards.values())]
        return list(rewards.values()), list(vm_rewards.values())

    def _get_done(self):
        return self.steps >= self.episode_limit

    def _get_info(self, info={}):
        demandloss = 0
        v = self.net.res_bus["vm_pu"].sort_index().to_numpy()

        # percentage of voltage out of control
        percent_of_v_out_of_control = (
            np.sum(v < self.v_lower) + np.sum(v > self.v_upper)
        ) / v.shape[0]
        info["percentage_of_v_out_of_control"] = percent_of_v_out_of_control

        # voltage violtation
        v_ref = 0.5 * (self.v_lower + self.v_upper)
        info["average_voltage_deviation"] = np.mean(np.abs(v - v_ref))
        info["average_voltage"] = np.mean(v)
        for k in self.agents:
            de = self.buildings[k].get_demandreward(self.net)
            demandloss += de
        info["demand_loss"] = demandloss

        # line loss
        line_loss = np.sum(self.net.res_line["pl_mw"])
        avg_line_loss = np.mean(self.net.res_line["pl_mw"])
        info["total_line_loss"] = line_loss

        # reactive power (q) loss
        q = self.net.res_sgen["q_mvar"].sort_index().to_numpy(copy=True)
        q_loss = np.mean(np.abs(q))
        info["q_loss"] = q_loss

        return info

    def _get_bus_clusters(self):
        # calc temp matrix that show adjacent bus
        G = np.eye(len(self.net.bus))
        id1 = self.net.line["from_bus"].tolist()
        id2 = self.net.line["to_bus"].tolist()
        G[id1, id2] = 1
        G[id2, id1] = 1
        temp = matrix_power(G, self.cluster_adjacent_bus_num)

        # clusters: BuildingsID --> adjacent bus BuildingsID
        clusters = dict()
        for agent in self.net.load["name"].tolist():
            agent_bus = self.net.load.loc[self.net.load["name"] == agent, "bus"]
            adjacent_bus = np.where(temp[agent_bus].squeeze() > 0)[0]
            clusters[agent] = self.net.load.loc[
                self.net.load["bus"].isin(adjacent_bus), "name"
            ].tolist()

        return clusters

    def _get_partial_obs_max_len(self):
        max_adjacent_agents = 0
        for adjacent_list in list(self.clusters.values()):
            if len(adjacent_list) > max_adjacent_agents:
                max_adjacent_agents = len(adjacent_list)

        return max_adjacent_agents * self.single_agent_obs_size

    def get_scheme(self):
        env_info = self.get_env_info()
        scheme = {
            "states": {"vshape": env_info["state_shape"], "dtype": np.float32},
            "obs": {
                "vshape": env_info["obs_shape"],
                "dtype": np.float32,
                "group": "agents",
            },
            "actions": {"vshape": (4,), "group": "agents", "dtype": np.float32},
            "available_actions": {
                "vshape": (4,),
                "group": "agents",
                "dtype": np.float32,
            },
            "rewards": {"vshape": (1,), "dtype": np.float32},
            "terminated": {"vshape": (1,), "dtype": np.int32},
        }
        groups = {"agents": env_info["n_agents"]}
        return scheme, groups

    def close(self):
        return True
