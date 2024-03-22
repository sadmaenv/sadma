import json
from math import sqrt
import os
import sys
import time

import cityflow as engine
import numpy as np
from smac.env import MultiAgentEnv
import random



def get_tpid():
    current_time = time.time()
    formatted_time = time.strftime("%Y%m%d-%H:%M:%S", time.localtime(current_time))
    return formatted_time + "-" + str(os.getpid())


class Intersection:
    def __init__(
        self, inter_name, dic_traffic_env_conf, eng, light_id_dict, lanes_length_dict
    ):
        self.inter_name = inter_name
        inter_id = (
            int(inter_name.split("_")[-2]),
            int(inter_name.split("_")[-1]),
        )  # 要求名称格式为 intersection_20_13，曼哈顿的格式就不对，100522479
        self.eng = eng
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.lane_length = lanes_length_dict

        self.list_approachs = ["W", "E", "N", "S"]
        # corresponding exiting lane for entering lanes
        self.dic_approach_to_node = {"W": 0, "E": 2, "S": 1, "N": 3}
        self.dic_entering_approach_to_edge = {
            "W": "road_{0}_{1}_0".format(inter_id[0] - 1, inter_id[1])
        }
        self.dic_entering_approach_to_edge.update(
            {"E": "road_{0}_{1}_2".format(inter_id[0] + 1, inter_id[1])}
        )
        self.dic_entering_approach_to_edge.update(
            {"N": "road_{0}_{1}_3".format(inter_id[0], inter_id[1] + 1)}
        )
        self.dic_entering_approach_to_edge.update(
            {"S": "road_{0}_{1}_1".format(inter_id[0], inter_id[1] - 1)}
        )
        self.dic_exiting_approach_to_edge = {
            approach: "road_{0}_{1}_{2}".format(
                inter_id[0], inter_id[1], self.dic_approach_to_node[approach]
            )
            for approach in self.list_approachs
        }
        self.list_phases = dic_traffic_env_conf["PHASE"]

        # generate all lanes
        self.list_entering_lanes = []
        for approach, lane_number in zip(
            self.list_approachs, dic_traffic_env_conf["NUM_LANES"]
        ):
            self.list_entering_lanes += [
                self.dic_entering_approach_to_edge[approach] + "_" + str(i)
                for i in range(lane_number)
            ]
        self.list_exiting_lanes = []
        for approach, lane_number in zip(
            self.list_approachs, dic_traffic_env_conf["NUM_LANES"]
        ):
            self.list_exiting_lanes += [
                self.dic_exiting_approach_to_edge[approach] + "_" + str(i)
                for i in range(lane_number)
            ]

        self.list_lanes = self.list_entering_lanes + self.list_exiting_lanes

        self.adjacency_row = light_id_dict["adjacency_row"]
        self.neighbor_ENWS = light_id_dict["neighbor_ENWS"]

        # ========== record previous & current feats ==========
        self.dic_lane_vehicle_previous_step = {}
        self.dic_lane_vehicle_previous_step_in = {}
        self.dic_lane_waiting_vehicle_count_previous_step = {}
        self.dic_vehicle_speed_previous_step = {}
        self.dic_vehicle_distance_previous_step = {}

        # in [entering_lanes] out [exiting_lanes]
        self.dic_lane_vehicle_current_step_in = {}
        self.dic_lane_vehicle_current_step = {}
        self.dic_lane_waiting_vehicle_count_current_step = {}
        self.dic_vehicle_speed_current_step = {}
        self.dic_vehicle_distance_current_step = {}

        self.list_lane_vehicle_previous_step_in = []
        self.list_lane_vehicle_current_step_in = []

        self.dic_vehicle_arrive_leave_time = dict()  # cumulative

        self.dic_feature = {}  # this second
        self.dic_feature_previous_step = {}  # this second

        # =========== signal info set ================
        # -1: all yellow, -2: all red, -3: none
        self.all_yellow_phase_index = -1
        self.all_red_phase_index = -2

        self.current_phase_index = 1
        self.previous_phase_index = 1
        self.eng.set_tl_phase(self.inter_name, self.current_phase_index)

        self.next_phase_to_set_index = None
        self.current_phase_duration = -1
        self.all_red_flag = False
        self.all_yellow_flag = False
        self.flicker = 0

    def set_signal(self, action, action_pattern, yellow_time):
        if self.all_yellow_flag:
            # in yellow phase
            self.flicker = 0
            if self.current_phase_duration >= yellow_time:  # yellow time reached
                self.current_phase_index = self.next_phase_to_set_index
                self.eng.set_tl_phase(
                    self.inter_name, self.current_phase_index
                )  # if multi_phase, need more adjustment

                self.all_yellow_flag = False
        else:
            # determine phase
            if action_pattern == "switch":  # switch by order
                if action == 0:  # keep the phase
                    self.next_phase_to_set_index = self.current_phase_index
                elif action == 1:  # change to the next phase
                    self.next_phase_to_set_index = (self.current_phase_index + 1) % len(
                        self.list_phases
                    )
                    # if multi_phase, need more adjustment
                else:
                    sys.exit("action not recognized\n action must be 0 or 1")

            elif action_pattern == "set":  # set to certain phase
                # self.next_phase_to_set_index = self.DIC_PHASE_MAP[action] # if multi_phase, need more adjustment
                self.next_phase_to_set_index = action + 1
            # set phase
            if self.current_phase_index == self.next_phase_to_set_index:
                # the light phase keeps unchanged
                pass
            else:  # the light phase needs to change
                # change to yellow first, and activate the counter and flag
                self.eng.set_tl_phase(self.inter_name, 0)  # !!! yellow, tmp
                self.current_phase_index = self.all_yellow_phase_index
                self.all_yellow_flag = True
                self.flicker = 1

    # update inner measurements
    def update_previous_measurements(self):
        self.previous_phase_index = self.current_phase_index
        self.dic_lane_vehicle_previous_step = self.dic_lane_vehicle_current_step
        self.dic_lane_vehicle_previous_step_in = self.dic_lane_vehicle_current_step_in
        self.dic_lane_waiting_vehicle_count_previous_step = (
            self.dic_lane_waiting_vehicle_count_current_step
        )
        self.dic_vehicle_speed_previous_step = self.dic_vehicle_speed_current_step
        self.dic_vehicle_distance_previous_step = self.dic_vehicle_distance_current_step

    def update_current_measurements(self, simulator_state):
        def _change_lane_vehicle_dic_to_list(dic_lane_vehicle):
            list_lane_vehicle = []
            for value in dic_lane_vehicle.values():
                list_lane_vehicle.extend(value)
            return list_lane_vehicle

        if self.current_phase_index == self.previous_phase_index:
            self.current_phase_duration += 1
        else:
            self.current_phase_duration = 1

        self.dic_lane_vehicle_current_step = {}
        self.dic_lane_vehicle_current_step_in = {}
        self.dic_lane_waiting_vehicle_count_current_step = {}
        for lane in self.list_entering_lanes:
            self.dic_lane_vehicle_current_step_in[lane] = simulator_state[
                "get_lane_vehicles"
            ][lane]

        for lane in self.list_lanes:
            self.dic_lane_vehicle_current_step[lane] = simulator_state[
                "get_lane_vehicles"
            ][lane]
            self.dic_lane_waiting_vehicle_count_current_step[lane] = simulator_state[
                "get_lane_waiting_vehicle_count"
            ][lane]

        self.dic_vehicle_speed_current_step = simulator_state["get_vehicle_speed"]
        self.dic_vehicle_distance_current_step = simulator_state["get_vehicle_distance"]

        # get vehicle list
        self.list_lane_vehicle_current_step_in = _change_lane_vehicle_dic_to_list(
            self.dic_lane_vehicle_current_step_in
        )
        self.list_lane_vehicle_previous_step_in = _change_lane_vehicle_dic_to_list(
            self.dic_lane_vehicle_previous_step_in
        )

        list_vehicle_new_arrive = list(
            set(self.list_lane_vehicle_current_step_in)
            - set(self.list_lane_vehicle_previous_step_in)
        )
        # can't use empty set to - real set
        if not self.list_lane_vehicle_previous_step_in:  # previous step is empty
            list_vehicle_new_left = list(
                set(self.list_lane_vehicle_current_step_in)
                - set(self.list_lane_vehicle_previous_step_in)
            )
        else:
            list_vehicle_new_left = list(
                set(self.list_lane_vehicle_previous_step_in)
                - set(self.list_lane_vehicle_current_step_in)
            )
        # update vehicle arrive and left time
        self._update_arrive_time(list_vehicle_new_arrive)
        self._update_left_time(list_vehicle_new_left)
        # update feature
        self._update_feature()

    def _update_leave_entering_approach_vehicle(self):
        list_entering_lane_vehicle_left = []
        # update vehicles leaving entering lane
        if not self.dic_lane_vehicle_previous_step:  # the dict is not empty
            for _ in self.list_entering_lanes:
                list_entering_lane_vehicle_left.append([])
        else:
            last_step_vehicle_id_list = []
            current_step_vehilce_id_list = []
            for lane in self.list_entering_lanes:
                last_step_vehicle_id_list.extend(
                    self.dic_lane_vehicle_previous_step[lane]
                )
                current_step_vehilce_id_list.extend(
                    self.dic_lane_vehicle_current_step[lane]
                )

            list_entering_lane_vehicle_left.append(
                list(set(last_step_vehicle_id_list) - set(current_step_vehilce_id_list))
            )
        return list_entering_lane_vehicle_left

    def _update_arrive_time(self, list_vehicle_arrive):
        ts = self.get_current_time()
        # get dic vehicle enter leave time
        for vehicle in list_vehicle_arrive:
            if vehicle not in self.dic_vehicle_arrive_leave_time:
                self.dic_vehicle_arrive_leave_time[vehicle] = {
                    "enter_time": ts,
                    "leave_time": np.nan,
                }

    def _update_left_time(self, list_vehicle_left):
        ts = self.get_current_time()
        # update the time for vehicle to leave entering lane
        for vehicle in list_vehicle_left:
            try:
                self.dic_vehicle_arrive_leave_time[vehicle]["leave_time"] = ts
            except KeyError:
                print("vehicle not recorded when entering")
                sys.exit(-1)

    def _update_feature(self):
        dic_feature = dict()
        dic_feature["cur_phase"] = [self.current_phase_index]
        dic_feature["time_this_phase"] = [self.current_phase_duration]
        dic_feature["lane_num_vehicle_in"] = self._get_lane_num_vehicle(
            self.list_entering_lanes
        )
        dic_feature["lane_num_vehicle_out"] = self._get_lane_num_vehicle(
            self.list_exiting_lanes
        )

        dic_feature["lane_queue_vehicle_in"] = self._get_lane_queue_length(
            self.list_entering_lanes
        )
        dic_feature["lane_queue_vehicle_out"] = self._get_lane_queue_length(
            self.list_exiting_lanes
        )

        dic_feature["traffic_movement_pressure_queue"] = (
            self._get_traffic_movement_pressure_general(
                dic_feature["lane_queue_vehicle_in"],
                dic_feature["lane_queue_vehicle_out"],
            )
        )

        dic_feature["traffic_movement_pressure_num"] = (
            self._get_traffic_movement_pressure_general(
                dic_feature["lane_num_vehicle_in"], dic_feature["lane_num_vehicle_out"]
            )
        )

        dic_feature["pressure"] = self._get_pressure(
            dic_feature["lane_queue_vehicle_in"], dic_feature["lane_queue_vehicle_out"]
        )
        dic_feature["adjacency_matrix"] = self._get_adjacency_row()
        self.dic_feature = dic_feature

    @staticmethod
    def _get_traffic_movement_pressure_general(enterings, exitings):
        """
        Created by LiangZhang
        Calculate pressure with entering and exiting vehicles
        only for 3 x 3 lanes intersection
        """
        list_approachs = ["W", "E", "N", "S"]
        index_maps = {"W": [0, 1, 2], "E": [3, 4, 5], "N": [6, 7, 8], "S": [9, 10, 11]}
        # vehicles in exiting road
        outs_maps = {}
        for approach in list_approachs:
            outs_maps[approach] = sum([exitings[i] for i in index_maps[approach]])
        turn_maps = ["S", "W", "N", "N", "E", "S", "W", "N", "E", "E", "S", "W"]
        t_m_p = [enterings[j] - outs_maps[turn_maps[j]] for j in range(12)]
        return t_m_p

    def _get_pressure(self, l_v_in, l_v_out):
        return list(np.array(l_v_in) - np.array(l_v_out))

    def _get_lane_queue_length(self, list_lanes):
        """
        queue length for each lane
        """
        return [
            self.dic_lane_waiting_vehicle_count_current_step[lane]
            for lane in list_lanes
        ]

    def _get_lane_num_vehicle(self, list_lanes):
        return [len(self.dic_lane_vehicle_current_step[lane]) for lane in list_lanes]

    # ================= get functions from outside ======================
    def get_current_time(self):
        return self.eng.get_current_time()

    def get_dic_vehicle_arrive_leave_time(self):
        return self.dic_vehicle_arrive_leave_time

    def get_feature(self):
        return self.dic_feature

    def get_state(self, list_state_features):
        dic_state = {
            state_feature_name: self.dic_feature[state_feature_name]
            for state_feature_name in list_state_features
        }
        return dic_state

    def _get_adjacency_row(self):
        return self.adjacency_row

    def get_reward(self, dic_reward_info):
        dic_reward = dict()
        # dic_reward["sum_lane_queue_length"] = None
        dic_reward["pressure"] = np.absolute(np.sum(self.dic_feature["pressure"]))
        dic_reward["queue_length"] = np.absolute(
            np.sum(self.dic_feature["lane_queue_vehicle_in"])
        )
        reward = 0
        for r in dic_reward_info:
            if dic_reward_info[r] != 0:
                reward += dic_reward_info[r] * dic_reward[r]
        return reward


def dict_value2array(dictkv):
    """将字典的值转换为列表

    >>> dict_value2array({'cur_phase': [1], 'lane_queue_vehicle_in': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2]})
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2]
    >>> dict_value2array({'cur_phase': [1], 'lane_queue_vehicle_in': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'intersection_2_1': {'cur_phase': [1], 'lane_queue_vehicle_in': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}, 'intersection_1_2': {'cur_phase': [1], 'lane_queue_vehicle_in': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}})
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    """
    values = []
    for v in dictkv.values():
        if isinstance(v, list):  # type(v)==list
            values += v
        elif isinstance(v, dict):  # type(v)==dict 字典嵌套
            values += dict_value2array(v)
        else:
            assert False, "Not list or dict type. type of v: " + str(v)
    return values


def dict_value_all2_target(dictkv, target=0):
    """将字典的值都设置为 target
    >>> dict_value_all2_target({'cur_phase': [1], 'lane_queue_vehicle_in': [0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 3, 0]})
    {'cur_phase': [0], 'lane_queue_vehicle_in': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
    """
    newdictkv = {}
    for k in dictkv.keys():
        v = dictkv[k]
        if isinstance(v, list):
            newdictkv[k] = [target for _ in range(len(v))]
        elif isinstance(v, dict):
            newdictkv[k] = dict_value_all2_target(v)
        else:
            assert False, "Not list or dict type. type of v: " + str(v)
    return newdictkv


def save_json(content, file_path):
    """将 json 数据保存到文件中"""
    with open(file_path, "w") as json_file:
        json.dump(content, json_file, indent=2)


class CityflowEnv(MultiAgentEnv):
    def __init__(self, **kwargs):
        rewrite_config = True  # 是否重写配置文件，在同一参数的实验中设置为 False 可以减少文件读写时间
        if "rewrite_config" in kwargs.keys():
            rewrite_config = kwargs["rewrite_config"]
        args_map_name = kwargs["map_name"].split("-")
        map_name = args_map_name[0]
        map_index = 0  # 每个城市车流数据个数不尽相同
        if len(args_map_name) > 1:
            map_index = (
                int(args_map_name[1]) - 1
            )  # -1 处理，若输入地图索引，便于从 1 开始
        if map_name == "jinan":
            roadnet_file = "cityflow_data/Jinan/3_4/roadnet_3_4.json"
            FLOW_FILE = [
                "cityflow_data/Jinan/3_4/anon_3_4_jinan_real.json",
                "cityflow_data/Jinan/3_4/anon_3_4_jinan_real_2000.json",
                "cityflow_data/Jinan/3_4/anon_3_4_jinan_real_2500.json",
            ]
        elif map_name == "hangzhou":
            roadnet_file = "cityflow_data/Hangzhou/4_4/roadnet_4_4.json"
            FLOW_FILE = [
                "cityflow_data/Hangzhou/4_4/anon_4_4_hangzhou_real.json",
                "cityflow_data/Hangzhou/4_4/anon_4_4_hangzhou_real_5816.json",
            ]
        elif map_name == "newyork":
            roadnet_file = "cityflow_data/newyork_28_7/28_7/roadnet_28_7.json"
            FLOW_FILE = [
                "cityflow_data/newyork_28_7/28_7/anon_28_7_newyork_real_double.json",
                "cityflow_data/newyork_28_7/28_7/anon_28_7_newyork_real_triple.json",
            ]
        elif map_name == "1225":
            roadnet_file = "cityflow_data/1225/roadnet_35_35.json"
            FLOW_FILE = [
                "cityflow_data/1225/flow_35_35.json",
            ]
        elif map_name == "33x34":
            roadnet_file = "cityflow_data/33_34/roadnet_33_34.json"
            FLOW_FILE = [
                "cityflow_data/33_34/flow_33_34.json",
            ]
        else:
            assert False, "map_name should be in [jinan, hangzhou, newyork]"
        flow_file = FLOW_FILE[map_index]
        self.path_to_work_directory = kwargs["log_path"]
        # 给 CityFLow 配置文件中的 dir 最后加上斜杠，不然 CityFLow 就会报错
        if not self.path_to_work_directory[-1] == "/":
            self.path_to_work_directory += "/"
        # 为数据文件夹创建软链接
        if not os.path.exists(
            os.path.join(self.path_to_work_directory, "cityflow_data")
        ):
            os.symlink(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "cityflow_data"
                ),
                os.path.join(self.path_to_work_directory, "cityflow_data"),
            )

        self.dic_traffic_env_conf = {
            "RUN_COUNTS": 3600,
            "TOP_K_ADJACENCY": 5,
            "ACTION_PATTERN": "set",
            "MIN_ACTION_TIME": 15,
            "YELLOW_TIME": kwargs["yellow_time"],
            "NUM_LANES": [3, 3, 3, 3],
            "LIST_STATE_FEATURE": kwargs["list_state_feature"],
            "DIC_REWARD_INFO": kwargs["dic_reward_info"],
            "PHASE": {
                1: [0, 1, 0, 1, 0, 0, 0, 0],
                2: [0, 0, 0, 0, 0, 1, 0, 1],
                3: [1, 0, 1, 0, 0, 0, 0, 0],
                4: [0, 0, 0, 0, 1, 0, 1, 0],
            },
            "list_lane_order": ["WL", "WT", "EL", "ET", "NL", "NT", "SL", "ST"],
            "PHASE_LIST": ["WT_ET", "NT_ST", "WL_EL", "NL_SL"],
            "INTERVAL": kwargs["interval"],
            "seed": kwargs["seed"],
            "laneChange": kwargs["lane_change"],
            "roadnetFile": roadnet_file,
            "flowFile": flow_file,
            "saveReplay": kwargs["save_replay"],
            "thread_num": kwargs["thread_num"],
        }
        eightphase = kwargs["eight_phase"]
        if eightphase:
            self.dic_traffic_env_conf["PHASE"] = {
                1: [0, 1, 0, 1, 0, 0, 0, 0],
                2: [0, 0, 0, 0, 0, 1, 0, 1],
                3: [1, 0, 1, 0, 0, 0, 0, 0],
                4: [0, 0, 0, 0, 1, 0, 1, 0],
                5: [1, 1, 0, 0, 0, 0, 0, 0],
                6: [0, 0, 1, 1, 0, 0, 0, 0],
                7: [0, 0, 0, 0, 0, 0, 1, 1],
                8: [0, 0, 0, 0, 1, 1, 0, 0],
            }
            self.dic_traffic_env_conf["PHASE_LIST"] = [
                "WT_ET",
                "NT_ST",
                "WL_EL",
                "NL_SL",
                "WL_WT",
                "EL_ET",
                "SL_ST",
                "NL_NT",
            ]
        self.current_time = None
        self.id_to_index = None
        self.traffic_light_node_dict = None
        self.eng = None
        self.list_intersection = None
        self.list_inter_log = None
        self.list_lanes = None  # TODO 似乎删除不影响
        self.system_states = None
        self.lane_length = None

        # check min action time
        if (
            self.dic_traffic_env_conf["MIN_ACTION_TIME"]
            <= self.dic_traffic_env_conf["YELLOW_TIME"]
        ):
            """include the yellow time in action time"""
            print("MIN_ACTION_TIME should include YELLOW_TIME")
            sys.exit()

        cityflow_config = {
            "interval": self.dic_traffic_env_conf["INTERVAL"],
            "seed": self.dic_traffic_env_conf["seed"],
            "dir": self.path_to_work_directory,
            "roadnetFile": self.dic_traffic_env_conf["roadnetFile"],
            "flowFile": self.dic_traffic_env_conf["flowFile"],
            "rlTrafficLight": True,
            "laneChange": self.dic_traffic_env_conf["laneChange"],
            "saveReplay": self.dic_traffic_env_conf["saveReplay"],
            "roadnetLogFile": "roadnetLogFile.json",
            "replayLogFile": "replayLogFile.txt",
        }
        self.cityflow_config_filepath = os.path.join(
            self.path_to_work_directory, "cityflowconfig.json"
        )
        if not os.path.exists(self.cityflow_config_filepath):
            save_json(cityflow_config, self.cityflow_config_filepath)
        else:  # 配置文件存在
            if rewrite_config:
                save_json(cityflow_config, self.cityflow_config_filepath)
        time.sleep(1)
        self.reset()

    def reset(self):
        self.step_num = 0
        self.eng = engine.Engine(
            self.cityflow_config_filepath,
            thread_num=self.dic_traffic_env_conf["thread_num"],
        )

        if self.dic_traffic_env_conf["saveReplay"]:
            # 避免不同线程将回放写入到同一个文件,roadnetLogFile 对于同一个地图是一样的，且没有后期改这个文件路径的 api(不需要)
            self.eng.set_replay_file(f"replayLogFile-{get_tpid()}.txt")

        # get adjacency
        self.traffic_light_node_dict = self._adjacency_extraction()

        # get lane length
        _, self.lane_length = self.get_lane_length()

        # initialize intersections (grid)
        self.list_intersection = {}
        self.list_inter_log = []
        for k, v in self.traffic_light_node_dict.items():
            self.list_intersection[k] = Intersection(
                k, self.dic_traffic_env_conf, self.eng, v, self.lane_length
            )
            self.list_inter_log.append([])

        self.n_agents = len(self.list_intersection.keys())
        self.episode_limit = self.dic_traffic_env_conf["RUN_COUNTS"]

        self.list_lanes = []
        for inter in self.list_intersection.values():
            self.list_lanes += inter.list_lanes
        self.list_lanes = np.unique(self.list_lanes).tolist()

        # get new measurements
        self.system_states = {
            "get_lane_vehicles": self.eng.get_lane_vehicles(),
            "get_lane_waiting_vehicle_count": self.eng.get_lane_waiting_vehicle_count(),
            "get_vehicle_speed": self.eng.get_vehicle_speed(),
            "get_vehicle_distance": self.eng.get_vehicle_distance(),
        }

        for inter in self.list_intersection.values():
            inter.update_current_measurements(self.system_states)
        state = self.get_state()
        return state

    def step(self, action):
        if (
            type(action[0]) == np.ndarray
        ):  # fix TypeError sync env_batch_size==1 TODO check
            action = np.squeeze(action)  # [[1], [2], [3]] -> [1 2 3]
        list_action_in_sec = [action]
        list_action_in_sec_display = [action]
        for i in range(self.dic_traffic_env_conf["MIN_ACTION_TIME"] - 1):
            if self.dic_traffic_env_conf["ACTION_PATTERN"] == "switch":
                list_action_in_sec.append(np.zeros_like(action).tolist())
            elif self.dic_traffic_env_conf["ACTION_PATTERN"] == "set":
                list_action_in_sec.append(np.copy(action).tolist())
            list_action_in_sec_display.append(
                np.full_like(action, fill_value=-1).tolist()
            )

        average_reward_action_list = [0] * len(action)
        for i in range(self.dic_traffic_env_conf["MIN_ACTION_TIME"]):
            action_in_sec = list_action_in_sec[i]
            action_in_sec_display = list_action_in_sec_display[i]

            instant_time = self.get_current_time()
            self.current_time = self.get_current_time()

            before_action_feature = self.get_feature()
            self._inner_step(action_in_sec)

            # get reward
            reward = self.get_reward()
            for j in range(len(reward)):
                average_reward_action_list[j] = (
                    average_reward_action_list[j] * i + reward[j]
                ) / (i + 1)
            self.log(
                cur_time=instant_time,
                before_action_feature=before_action_feature,
                action=action_in_sec_display,
            )
        info = {"step": self.current_time}
        # info["individual_rewards"]= np.array(reward).astype(np.float32) / 1e4

        done = False
        if self.step_num > int(
            self.dic_traffic_env_conf["RUN_COUNTS"]
            / self.dic_traffic_env_conf["MIN_ACTION_TIME"]
        ):
            done = True
            att = self.get_avg_travel_time()
            info["average travel time"] = att
            info["total reward"] = np.sum(reward)
        self.step_num += 1
        return np.sum(reward), done, info

    def _inner_step(self, action):
        # copy current measurements to previous measurements
        for inter in self.list_intersection.values():
            inter.update_previous_measurements()
        # set signals
        # multi_intersection decided by action {inter_id: phase}
        for inter_ind, inter in enumerate(self.list_intersection.values()):
            inter.set_signal(
                action=action[inter_ind],
                action_pattern=self.dic_traffic_env_conf["ACTION_PATTERN"],
                yellow_time=self.dic_traffic_env_conf["YELLOW_TIME"],
            )

        # run one step
        for i in range(int(1 / self.dic_traffic_env_conf["INTERVAL"])):
            self.eng.next_step()

        self.system_states = {
            "get_lane_vehicles": self.eng.get_lane_vehicles(),
            "get_lane_waiting_vehicle_count": self.eng.get_lane_waiting_vehicle_count(),
            "get_vehicle_speed": self.eng.get_vehicle_speed(),
            "get_vehicle_distance": self.eng.get_vehicle_distance(),
        }

        for inter in self.list_intersection.values():
            inter.update_current_measurements(self.system_states)

    def get_feature(self):
        list_feature = [
            inter.get_feature() for inter in self.list_intersection.values()
        ]
        return list_feature

    def get_state_dict(self, near_state=True):
        """将状态以字典形式返回
        near_state True时则会返回其相邻路口的状态
        """
        dict_state = []
        for iname in self.list_intersection.keys():
            inter = self.list_intersection[iname]
            inter_state = inter.get_state(
                self.dic_traffic_env_conf["LIST_STATE_FEATURE"]
            )
            if near_state:
                for index, neighbor_inter_name in enumerate(inter.neighbor_ENWS):
                    if neighbor_inter_name:  # maybe None
                        # 补充相邻非虚拟路口的obs
                        i_state = self.list_intersection[neighbor_inter_name].get_state(
                            self.dic_traffic_env_conf["LIST_STATE_FEATURE"]
                        )
                        inter_state[neighbor_inter_name] = i_state
                    else:  # None 也要补全，通常是虚拟的边缘路口
                        # 用自己路口的数据填充，然后全填0。虚拟路口没有相位，但有车流数据
                        i_state = inter.get_state(
                            self.dic_traffic_env_conf["LIST_STATE_FEATURE"]
                        )
                        inter_state[str(neighbor_inter_name) + str(index)] = (
                            dict_value_all2_target(i_state)
                        )
            dict_state.append(inter_state)
        return dict_state

    def get_state(self, near_state=False, with_att=False):
        dict_state = self.get_state_dict(near_state)
        list_state = []
        for di in dict_state:
            list_state+=dict_value2array(di)
            # list_state.append(dict_value2array(di))
        if with_att:
            list_state.append(self.get_avg_travel_time())
        return list_state

    def get_reward(self):
        list_reward = [
            inter.get_reward(self.dic_traffic_env_conf["DIC_REWARD_INFO"])
            for inter in self.list_intersection.values()
        ]
        return list_reward

    def get_current_time(self):
        return self.eng.get_current_time()

    def log(self, cur_time, before_action_feature, action):
        for inter_ind in range(len(self.list_intersection)):
            self.list_inter_log[inter_ind].append(
                {
                    "time": cur_time,
                    "state": before_action_feature[inter_ind],
                    "action": action[inter_ind],
                }
            )

    def _adjacency_extraction(self):
        traffic_light_node_dict = {}
        file = self.get_roadnet_file_path()

        with open("{0}".format(file)) as json_data:
            net = json.load(json_data)
            for inter in net["intersections"]:
                if not inter["virtual"]:
                    traffic_light_node_dict[inter["id"]] = {
                        "location": {
                            "x": float(inter["point"]["x"]),
                            "y": float(inter["point"]["y"]),
                        },
                        "total_inter_num": None,
                        "adjacency_row": None,
                        "inter_id_to_index": None,
                        "neighbor_ENWS": None,
                    }

            top_k = self.dic_traffic_env_conf["TOP_K_ADJACENCY"]
            total_inter_num = len(traffic_light_node_dict.keys())
            inter_id_to_index = {}

            edge_id_dict = {}
            for road in net["roads"]:
                if road["id"] not in edge_id_dict.keys():
                    edge_id_dict[road["id"]] = {}
                edge_id_dict[road["id"]]["from"] = road["startIntersection"]
                edge_id_dict[road["id"]]["to"] = road["endIntersection"]

            index = 0
            for i in traffic_light_node_dict.keys():
                inter_id_to_index[i] = index
                index += 1

            for i in traffic_light_node_dict.keys():
                location_1 = traffic_light_node_dict[i]["location"]

                row = np.array([0] * total_inter_num)
                for j in traffic_light_node_dict.keys():
                    location_2 = traffic_light_node_dict[j]["location"]
                    dist = self._cal_distance(location_1, location_2)
                    row[inter_id_to_index[j]] = dist
                if len(row) == top_k:
                    adjacency_row_unsorted = np.argpartition(row, -1)[:top_k].tolist()
                elif len(row) > top_k:
                    adjacency_row_unsorted = np.argpartition(row, top_k)[
                        :top_k
                    ].tolist()
                else:
                    adjacency_row_unsorted = [k for k in range(total_inter_num)]
                adjacency_row_unsorted.remove(inter_id_to_index[i])
                traffic_light_node_dict[i]["adjacency_row"] = [
                    inter_id_to_index[i]
                ] + adjacency_row_unsorted
                traffic_light_node_dict[i]["total_inter_num"] = total_inter_num

            for i in traffic_light_node_dict.keys():
                traffic_light_node_dict[i]["total_inter_num"] = inter_id_to_index
                traffic_light_node_dict[i]["neighbor_ENWS"] = []
                for j in range(4):
                    road_id = i.replace("intersection", "road") + "_" + str(j)
                    if (
                        edge_id_dict[road_id]["to"]
                        not in traffic_light_node_dict.keys()
                    ):
                        traffic_light_node_dict[i]["neighbor_ENWS"].append(None)
                    else:
                        traffic_light_node_dict[i]["neighbor_ENWS"].append(
                            edge_id_dict[road_id]["to"]
                        )

        return traffic_light_node_dict

    @staticmethod
    def _cal_distance(loc_dict1, loc_dict2):
        a = np.array((loc_dict1["x"], loc_dict1["y"]))
        b = np.array((loc_dict2["x"], loc_dict2["y"]))
        return np.sqrt(np.sum((a - b) ** 2))

    @staticmethod
    def end_cityflow():
        print("============== cityflow process end ===============")

    def get_avg_travel_time(self):
        """
        get average travel time
        """
        return self.eng.get_average_travel_time()

    def close(self):
        pass

    def get_lane_length(self):
        """
        newly added part for get lane length
        Read the road net file
        Return: dict{lanes} normalized with the min lane length
        """
        file = self.get_roadnet_file_path()
        with open(file) as json_data:
            net = json.load(json_data)
        roads = net["roads"]
        lanes_length_dict = {}
        lane_normalize_factor = {}

        for road in roads:
            points = road["points"]
            road_length = sqrt(
                (points[0]["x"] - points[1]["x"]) ** 2
                + (points[0]["y"] - points[1]["y"]) ** 2
            )
            assert road_length > 0, (
                "Error road length("
                + str(road_length)
                + ") should be bigger than 0. road info: "
                + str(road)
            )
            for i in range(3):
                lane_id = road["id"] + "_{0}".format(i)
                lanes_length_dict[lane_id] = road_length
        min_length = min(lanes_length_dict.values())

        for key, value in lanes_length_dict.items():
            lane_normalize_factor[key] = value / min_length
        return lane_normalize_factor, lanes_length_dict

    def get_roadnet_file_path(self):
        return os.path.join(
            self.path_to_work_directory, self.dic_traffic_env_conf["roadnetFile"]
        )

    def get_state_size(self):
        """Returns the shape of the state"""
        return len(self.get_state())

    def get_obs_size(self):
        """Returns the shape of the observation"""
        state = np.array(self.get_state(True, False))
        return state.size // self.n_agents

    def get_obs(self):
        state = np.array(self.get_state(True, False))
        return state.reshape(self.n_agents, state.size // self.n_agents)

    def get_total_actions(self):
        return len(self.dic_traffic_env_conf["PHASE"])

    def get_avail_actions(self):
        avail_actions = []
        for _ in range(self.n_agents):
            avail_agent = [1] * self.get_total_actions()
            avail_actions.append(avail_agent)
        return np.array(avail_actions)

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
