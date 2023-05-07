import os
import copy
import time
import gym
import numpy as np
from gym import spaces
from collections import defaultdict
from typing import Union, Dict, AnyStr, Tuple, Optional
from gym.envs.registration import register
import logging

from asaprl.utils.env_utils.discrete_policy import DiscreteMetaAction
from asaprl.utils.env_utils.agent_manager_utils import MacroAgentManager
from asaprl.utils.env_utils.engine_utils import initialize_engine, close_engine, \
    engine_initialized, set_global_random_seed, MacroBaseEngine
from asaprl.utils.env_utils.traffic_manager_utils import TrafficMode
from metadrive.constants import RENDER_MODE_NONE, DEFAULT_AGENT, REPLAY_DONE
from metadrive.envs.base_env import BaseEnv
from metadrive.component.map.base_map import BaseMap
from metadrive.component.map.pg_map import parse_map_config, MapGenerateMethod
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.constants import DEFAULT_AGENT, TerminationState
from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.utils import Config, merge_dicts, get_np_random, clip
from metadrive.utils import Config, merge_dicts, get_np_random, concat_step_infos
from metadrive.envs.base_env import BASE_DEFAULT_CONFIG
from metadrive.obs.top_down_obs_multi_channel import TopDownMultiChannel
from metadrive.utils.utils import auto_termination
from asaprl.policy.traj_vae import WpDecoder
from asaprl.policy.traj_vae import CCDecoder
import torch

METADRIVE_DEFAULT_CONFIG = dict(
    # ===== Generalization =====
    start_seed=0,
    use_render=False,
    environment_num=1,

    # ===== Map Config =====
    map='SSSSSSSSSS',  # int or string: an easy way to fill map_config
    random_lane_width=True,
    random_lane_num=True,
    map_config={
        BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_SEQUENCE,
        BaseMap.GENERATE_CONFIG: 'SSSSSSSSSS',  # None,  # it can be a file path / block num / block ID sequence
        BaseMap.LANE_WIDTH: 3.5,
        BaseMap.LANE_NUM: 3,
        "exit_length": 70,
    },

    # ===== Traffic =====
    traffic_density=0.0,
    on_screen=False,
    rgb_clip=True,
    need_inverse_traffic=False,
    traffic_mode=TrafficMode.Synch,  # "Respawn", "Trigger"
    random_traffic=True,  # Traffic is randomized at default.
    # this will update the vehicle_config and set to traffic
    traffic_vehicle_config=dict(
        show_navi_mark=False,
        show_dest_mark=False,
        enable_reverse=False,
        show_lidar=False,
        show_lane_line_detector=False,
        show_side_detector=False,
    ),

    # ===== Object =====
    accident_prob=0.,  # accident may happen on each block with this probability, except multi-exits block

    # ===== Others =====
    use_AI_protector=False,
    save_level=0.5,
    is_multi_agent=False,
    vehicle_config=dict(spawn_lane_index=(FirstPGBlock.NODE_1, FirstPGBlock.NODE_2, 0)),

    # ===== Agent =====
    target_vehicle_configs={
        DEFAULT_AGENT: dict(use_special_color=True, spawn_lane_index=(FirstPGBlock.NODE_1, FirstPGBlock.NODE_2, 2), overtake_stat=True)
    },

    # ===== Reward Scheme =====
    # See: https://github.com/decisionforce/metadrive/issues/283
    success_reward=10.0,
    out_of_road_penalty=5.0,
    crash_vehicle_penalty=1.0,
    crash_object_penalty=5.0,
    run_out_of_time_penalty = 5.0,
    driving_reward= 0.2,  #1.0,
    speed_reward=0.05,
    heading_reward = 0.15, 
    use_lateral=True,
    dense_reward = False,
    reward_average_length = False,
    
    # ===== Cost Scheme =====
    crash_vehicle_cost=1.0,
    crash_object_cost=1.0,
    out_of_road_cost=1.0,

    # ===== Termination Scheme =====
    out_of_route_done=True,
    physics_world_step_size=1e-1,

    # ===== Trajectory length =====
    seq_traj_len = 10,
    show_seq_traj = False,

    use_jerk_penalty = False,
    use_lateral_penalty = False,
    const_control = False,
    half_jerk = False,

    speed_to_cal_time_limit = 4,
    speed_limit_to_get_reward = 4.5

)


class MetaDriveControlEnv(BaseEnv):

    @classmethod
    def default_config(cls) -> "Config":
        #config = super(SimpleMetaDriveEnv, cls).default_config()
        config = Config(BASE_DEFAULT_CONFIG)
        config.update(METADRIVE_DEFAULT_CONFIG)
        config.register_type("map", str, int)
        config["map_config"].register_type("config", None)
        return config

    def __init__(self, config: dict = None):
        merged_config = self._merge_extra_config(config)
        global_config = self._post_process_config(merged_config)
        self.config = global_config

        # agent check
        self.num_agents = self.config["num_agents"]
        self.is_multi_agent = self.config["is_multi_agent"]
        if not self.is_multi_agent:
            assert self.num_agents == 1
        assert isinstance(self.num_agents, int) and (self.num_agents > 0 or self.num_agents == -1)

        # observation and action space
        self.agent_manager = MacroAgentManager(
            init_observations=self._get_observations(), init_action_space=self._get_action_space()
        )
        self.action_type = DiscreteMetaAction()
        #self.action_space = self.action_type.space()

        # lazy initialization, create the main vehicle in the lazy_init() func
        self.engine: Optional[MacroBaseEngine] = None
        self._top_down_renderer = None
        self.episode_steps = 0
        # self.current_seed = None

        # In MARL envs with respawn mechanism, varying episode lengths might happen.
        self.dones = None
        self.episode_rewards = defaultdict(float)
        self.episode_lengths = defaultdict(int)

        self.start_seed = self.config["start_seed"]
        self.env_num = self.config["environment_num"]

        self.time = 0
        self.step_num = 0
        self.episode_rwd = 0
        self.last_takeover_num = 0

        self.last_on_broken_line = 0
        self.last_finished_10m_road_num = 0
        self.env_step_num = 0
        self.step_traj_length = 1

        if not self.config["const_control"]:
            self.vae_decoder = WpDecoder(
                control_num = 2,
                seq_len = self.config['seq_traj_len'],
                dt = 0.1
                )
        else:
            self.vae_decoder = CCDecoder(
                control_num = 2,
                seq_len = self.config['seq_traj_len'],
                dt = 0.1
            )
        self.vel_speed = 0.0

    # define a action type, and execution style
    # Now only one action will be taken, cosin function, and we set dt equals self.engine.dt
    # now that in this situation, we directly set trajectory len equals to simulation frequency

    def step(self, actions: Union[np.ndarray, Dict[AnyStr, np.ndarray]]):
        self.episode_steps += 1
        init_state = np.zeros([1, 4])
        init_state[0,3] = self.vel_speed
        init_state = torch.from_numpy(init_state)
        if isinstance(actions, np.ndarray):
            batch_action = torch.from_numpy(actions)
            batch_action = torch.unsqueeze(batch_action, 0)
            batch_action = batch_action.to(torch.float32)
            init_state = init_state.to(torch.float32)
            with torch.no_grad():
                trajs = self.vae_decoder(batch_action, init_state)
            trajs = torch.cat([init_state.unsqueeze(1), trajs], dim = 1)
            trajs = trajs[:,:,:2]
            trajs = torch.squeeze(trajs, 0)
            actions = trajs.numpy()
        macro_actions = self._preprocess_macro_waypoints(actions)
        step_infos, self.step_traj_length = self._step_macro_simulator(macro_actions)
        self.env_step_num += self.step_traj_length # modify finished, distinguish between skill step and env step
        o, r, d, i = self._get_step_return(actions, step_infos)
        self.step_num = self.step_num + 1
        self.episode_rwd = self.episode_rwd + r 
        i['skill_horizon'] = self.config['seq_traj_len']

        return o, r, d, i

    def get_waypoint_list(self):
        x = np.arange(0, 6.2, 0.2)
        LENGTH = 0 # 4.51
        y = 1 * np.cos(np.pi*2 / 6.0 * x)-1
        x = x + LENGTH/2
        lst = []
        for i in range(x.shape[0]):
            lst.append([x[i],y[i]])
        return lst

    def _merge_extra_config(self, config: Union[dict, "Config"]) -> "Config":
        config = self.default_config().update(config, allow_add_new_key=True)
        if config["vehicle_config"]["lidar"]["distance"] > 50:
            config["max_distance"] = config["vehicle_config"]["lidar"]["distance"]
        return config

    def _post_process_config(self, config):
        config = super(MetaDriveControlEnv, self)._post_process_config(config)
        if not config["rgb_clip"]:
            logging.warning(
                "You have set rgb_clip = False, which means the observation will be uint8 values in [0, 255]. "
                "Please make sure you have parsed them later before feeding them to network!"
            )
        config["map_config"] = parse_map_config(
            easy_map_config=config["map"], new_map_config=config["map_config"], default_config=self.default_config()
        )
        config["vehicle_config"]["rgb_clip"] = config["rgb_clip"]
        config["vehicle_config"]["random_agent_model"] = config["random_agent_model"]
        if config.get("gaussian_noise", 0) > 0:
            assert config["vehicle_config"]["lidar"]["gaussian_noise"] == 0, "You already provide config!"
            assert config["vehicle_config"]["side_detector"]["gaussian_noise"] == 0, "You already provide config!"
            assert config["vehicle_config"]["lane_line_detector"]["gaussian_noise"] == 0, "You already provide config!"
            config["vehicle_config"]["lidar"]["gaussian_noise"] = config["gaussian_noise"]
            config["vehicle_config"]["side_detector"]["gaussian_noise"] = config["gaussian_noise"]
            config["vehicle_config"]["lane_line_detector"]["gaussian_noise"] = config["gaussian_noise"]
        if config.get("dropout_prob", 0) > 0:
            assert config["vehicle_config"]["lidar"]["dropout_prob"] == 0, "You already provide config!"
            assert config["vehicle_config"]["side_detector"]["dropout_prob"] == 0, "You already provide config!"
            assert config["vehicle_config"]["lane_line_detector"]["dropout_prob"] == 0, "You already provide config!"
            config["vehicle_config"]["lidar"]["dropout_prob"] = config["dropout_prob"]
            config["vehicle_config"]["side_detector"]["dropout_prob"] = config["dropout_prob"]
            config["vehicle_config"]["lane_line_detector"]["dropout_prob"] = config["dropout_prob"]
        target_v_config = copy.deepcopy(config["vehicle_config"])
        if not config["is_multi_agent"]:
            target_v_config.update(config["target_vehicle_configs"][DEFAULT_AGENT])
            config["target_vehicle_configs"][DEFAULT_AGENT] = target_v_config
        return config

    def _get_observations(self):
        return {DEFAULT_AGENT: self.get_single_observation(self.config["vehicle_config"])}

    def done_function(self, vehicle_id: str):
        vehicle = self.vehicles[vehicle_id]
        done = False
        done_info = dict(
            crash_vehicle=False, crash_object=False, crash_building=False, out_of_road=False, arrive_dest=False
        )
        if vehicle.arrive_destination:
            done = True
            logging.info("Episode ended! Reason: arrive_dest.")
            done_info[TerminationState.SUCCESS] = True
        elif hasattr(vehicle, 'macro_succ') and vehicle.macro_succ:
            done = True
            logging.info("Episode ended! Reason: arrive_dest.")
            done_info[TerminationState.SUCCESS] = True
        elif hasattr(vehicle, 'macro_crash') and vehicle.macro_crash:
            done = True
            logging.info("Episode ended! Reason: crash vehicle ")
            done_info[TerminationState.CRASH_VEHICLE] = True
        if self._is_out_of_road(vehicle):
            done = True
            logging.info("Episode ended! Reason: out_of_road.")
            done_info[TerminationState.OUT_OF_ROAD] = True
        if vehicle.crash_vehicle:
            done = True
            logging.info("Episode ended! Reason: crash vehicle ")
            done_info[TerminationState.CRASH_VEHICLE] = True
        if vehicle.crash_object:
            done = True
            done_info[TerminationState.CRASH_OBJECT] = True
            logging.info("Episode ended! Reason: crash object ")
        if vehicle.crash_building:
            done = True
            done_info[TerminationState.CRASH_BUILDING] = True
            logging.info("Episode ended! Reason: crash building ")
        if self.env_step_num > self.navi_distance / self.config['speed_to_cal_time_limit'] * 10:
            done = True
            done_info[TerminationState.MAX_STEP] = True
            logging.info("Episode ended! Reason: max env step reached")

        # for compatibility
        # crash almost equals to crashing with vehicles
        done_info[TerminationState.CRASH] = (
            done_info[TerminationState.CRASH_VEHICLE] or done_info[TerminationState.CRASH_OBJECT]
            or done_info[TerminationState.CRASH_BUILDING]
        )
        done_info['complete_ratio'] = clip(self.already_go_dist/ self.navi_distance + 0.05, 0.0, 1.0)

        return done, done_info

    def cost_function(self, vehicle_id: str):
        vehicle = self.vehicles[vehicle_id]
        step_info = dict()
        step_info["cost"] = 0
        if self._is_out_of_road(vehicle):
            step_info["cost"] = self.config["out_of_road_cost"]
        elif vehicle.crash_vehicle:
            step_info["cost"] = self.config["crash_vehicle_cost"]
        elif vehicle.crash_object:
            step_info["cost"] = self.config["crash_object_cost"]
        elif self.env_step_num > self.navi_distance / self.config['speed_to_cal_time_limit'] * 10:
            step_info['cost'] = 1
        return step_info['cost'], step_info

    def _is_out_of_road(self, vehicle):
        # A specified function to determine whether this vehicle should be done.
        # return vehicle.on_yellow_continuous_line or (not vehicle.on_lane) or vehicle.crash_sidewalk
        ret = vehicle.on_yellow_continuous_line or vehicle.on_white_continuous_line or \
              (not vehicle.on_lane) or vehicle.crash_sidewalk
        if self.config["out_of_route_done"]:
            ret = ret or vehicle.out_of_route
        return ret

    
    def reward_function(self, vehicle_id: str): 
        step_info = dict()
        reward = 0.0
        vehicle = self.vehicles[vehicle_id]
            
        ''' sparse '''
        # passing car + | passed by car -
        current_takeover_num = vehicle.get_overtake_num()
        reward += (current_takeover_num - self.last_takeover_num) * self.config['reward_w_pass_car']
        step_info["step_reward_pass_car"] = (current_takeover_num - self.last_takeover_num) * self.config['reward_w_pass_car']
        self.last_takeover_num = current_takeover_num
        
        # on lane: 
        reward += self.config['reward_w_on_lane'] * self.last_on_broken_line
        step_info["step_reward_on_broken_lane"] = self.config['reward_w_on_lane'] * self.last_on_broken_line
        self.last_on_broken_line = 0

        # out of road -
        if self._is_out_of_road(vehicle):
            reward += self.config['reward_w_out_of_road']
            step_info["step_reward_out_of_road"] = self.config['reward_w_out_of_road']
        else:
            step_info["step_reward_out_of_road"] = 0

        # crash_penalty -
        if vehicle.crash_vehicle or vehicle.macro_crash or vehicle.crash_object:
            reward += self.config['reward_w_crash']
            step_info["step_reward_crash"] = self.config['reward_w_crash']
        else:
            step_info["step_reward_crash"] = 0

        # arrive at destination +
        if vehicle.arrive_destination or vehicle.macro_succ:
            reward += self.config['reward_w_destination']
            step_info["step_reward_arrive_desti"] = self.config['reward_w_destination']
        else:
            step_info["step_reward_arrive_desti"] = 0

        # run out of time - 
        if self.env_step_num > self.navi_distance / self.config['speed_to_cal_time_limit'] * 10:
            reward += self.config['reward_w_out_of_time']
            step_info["step_reward_out_of_time"] = self.config['reward_w_out_of_time']
        else:
            step_info["step_reward_out_of_time"] = 0

        # progress reward +
        from math import floor
        if self._compute_navi_dist: 
            self.navi_distance = self.get_navigation_len(vehicle)
            self._compute_navi_dist = False
        if vehicle.lane in vehicle.navigation.current_ref_lanes:
            current_lane = vehicle.lane
            positive_road = 1
        else:
            current_lane = vehicle.navigation.current_ref_lanes[0]
            current_road = vehicle.navigation.current_road
            positive_road = 1 if not current_road.is_negative_road() else -1
        long_last, _ = current_lane.local_coordinates(vehicle.last_macro_position)
        long_now, lateral_now = current_lane.local_coordinates(vehicle.position)
        self.already_go_dist += (long_now - long_last)

        ''' 10 m land mark '''
        current_finished_10m_road_num = floor((self.already_go_dist/ 10))
        if self.last_finished_10m_road_num < current_finished_10m_road_num:
            reward += self.config['reward_w_progress']
            step_info["step_reward_complete_road"] = self.config['reward_w_progress']
        elif self.last_finished_10m_road_num > current_finished_10m_road_num:
            reward += - self.config['reward_w_progress']
            step_info["step_reward_complete_road"] = - self.config['reward_w_progress']
        else:
            step_info["step_reward_complete_road"] = 0
        self.last_finished_10m_road_num = current_finished_10m_road_num

        ''' speed reward '''
        speed_list = self.compute_speed_list(vehicle)
        ave_speed = sum(speed_list)/len(speed_list)
        if ave_speed >= self.config['speed_limit_to_get_reward']: 
            step_info["step_reward_speed"] = self.config['reward_w_speed']

        step_info["vehicle_last_speed"] = vehicle.last_spd

        if self.config['reward_average_length']: reward = reward / self.current_skill_length
        
        step_info["step_reward"] = reward
        return reward, step_info

    def reward_function_dense(self, vehicle_id: str):
        """
        Override this func to get a new reward function
        :param vehicle_id: id of BaseVehicle
        :return: reward
        """
        vehicle = self.vehicles[vehicle_id]
        if self._compute_navi_dist:
            self.navi_distance = self.get_navigation_len(vehicle)
            self._compute_navi_dist = False
        
        step_info = dict()

        # Reward for moving forward in current lane
        if vehicle.lane in vehicle.navigation.current_ref_lanes:
            current_lane = vehicle.lane
            positive_road = 1
        else:
            current_lane = vehicle.navigation.current_ref_lanes[0]
            current_road = vehicle.navigation.current_road
            positive_road = 1 if not current_road.is_negative_road() else -1
        long_last, _ = current_lane.local_coordinates(vehicle.last_macro_position)
        long_now, lateral_now = current_lane.local_coordinates(vehicle.position)
        self.already_go_dist += (long_now - long_last)

        avg_lateral_cum = self.compute_avg_lateral_cum(vehicle, current_lane)
        use_lateral_penalty = False
        # reward for lane keeping, without it vehicle can learn to overtake but fail to keep in lane
        if self.config["use_lateral"]:
            lateral_factor = clip(1 - 0.5 * abs(avg_lateral_cum) / vehicle.navigation.get_current_lane_width(), 0.0, 1.0)
        else:
            lateral_factor = 1.0
            use_lateral_penalty = True

        reward = 0.0
        # Driving reward   
        reward += self.config["driving_reward"] * (long_now - long_last) * lateral_factor * positive_road 

        # Speed reward
        max_spd = 10
        speed_list = self.compute_speed_list(vehicle)
        for speed in speed_list: 
            reward += self.config["speed_reward"] * (speed / max_spd) * positive_road     
            if speed < 4:
                reward -= 0.04

        # Heading Reward
        heading_error_list = self.compute_heading_error_list(vehicle, current_lane)
        for heading_error in heading_error_list:
            reward += self.config["heading_reward"] * (0 - np.abs(heading_error))    

        if self.config["use_jerk_penalty"]:
            jerk_list = self.compute_jerk_list(vehicle)
            for jerk in jerk_list:
                if not self.config["half_jerk"]:
                    reward += (0.03 - 0.6 * np.tanh(jerk / 100.0))
                else:
                    reward += (0.03 - 0.3 * np.tanh(jerk / 100.0))
        if use_lateral_penalty:
            lateral_penalty = avg_lateral_cum
            reward -= lateral_penalty /4 * 0.3
        step_info["step_reward"] = reward

        step_info["vehicle_last_speed"] = vehicle.last_spd
        current_takeover_num = vehicle.get_overtake_num()
        reward += (current_takeover_num - self.last_takeover_num) * self.config['reward_w_pass_car']
        step_info["step_reward_pass_car"] = (current_takeover_num - self.last_takeover_num) * 0.1
        self.last_takeover_num = current_takeover_num

        if vehicle.arrive_destination:
            reward = +self.config["success_reward"]
        elif vehicle.macro_succ:
            reward = +self.config["success_reward"]
        elif self._is_out_of_road(vehicle):
            reward = -self.config["out_of_road_penalty"]
        elif vehicle.crash_vehicle:
            reward = -self.config["crash_vehicle_penalty"]
        elif vehicle.macro_crash:
            reward = -self.config["crash_vehicle_penalty"]
        elif vehicle.crash_object:
            reward = -self.config["crash_object_penalty"]
        elif self.env_step_num > self.navi_distance / self.config['speed_to_cal_time_limit'] * 10:
            reward = - self.config["run_out_of_time_penalty"]
        return reward, step_info

    def get_navigation_len(self, vehicle):
        checkpoints = vehicle.navigation.checkpoints
        road_network = vehicle.navigation.map.road_network
        total_dist = 0
        assert len(checkpoints) >=2
        for check_num in range(0, len(checkpoints)-1):
            front_node = checkpoints[check_num]
            end_node = checkpoints[check_num+1] 
            cur_lanes = road_network.graph[front_node][end_node]
            target_lane_num = int(len(cur_lanes) / 2)
            target_lane = cur_lanes[target_lane_num]
            target_lane_length = target_lane.length
            total_dist += target_lane_length 
        return total_dist

    def compute_jerk_list(self, vehicle):
        jerk_list = []
        v_t0 = vehicle.penultimate_state['speed']
        theta_t0 = vehicle.penultimate_state['yaw']
        v_t1 = vehicle.traj_wp_list[0]['speed']
        theta_t1 = vehicle.traj_wp_list[0]['yaw']
        v_t2 = vehicle.traj_wp_list[1]['speed']
        theta_t2 = vehicle.traj_wp_list[1]['yaw']
        t_inverse = 1.0 / self.config['physics_world_step_size']
        first_point_jerk_x = (v_t2* np.cos(theta_t2) - 2 * v_t1 * np.cos(theta_t1) +  v_t0 * np.cos(theta_t0)) * t_inverse * t_inverse
        first_point_jerk_y = (v_t2* np.sin(theta_t2) - 2 * v_t1 * np.sin(theta_t1) +  v_t0 * np.sin(theta_t0)) * t_inverse * t_inverse
        jerk_list.append(np.array([first_point_jerk_x, first_point_jerk_y]))
        # plus one because we store the current value as first, which means the whole trajectory is seq_traj_len + 1
        for i in range(2, self.config['seq_traj_len'] + 1):
            v_t0 = vehicle.traj_wp_list[i-2]['speed']
            theta_t0 = vehicle.traj_wp_list[i-2]['yaw']
            v_t1 = vehicle.traj_wp_list[i-1]['speed']
            theta_t1 = vehicle.traj_wp_list[i-1]['yaw']
            v_t2 = vehicle.traj_wp_list[i]['speed']
            theta_t2 = vehicle.traj_wp_list[i]['yaw']    
            point_jerk_x = (v_t2* np.cos(theta_t2) - 2 * v_t1 * np.cos(theta_t1) + v_t0 * np.cos(theta_t0)) * t_inverse * t_inverse
            point_jerk_y = (v_t2* np.sin(theta_t2) - 2 * v_t1 * np.sin(theta_t1) + v_t0 * np.sin(theta_t0)) * t_inverse * t_inverse
            jerk_list.append(np.array([point_jerk_x, point_jerk_y]))
        #final_jerk_value = 0
        step_jerk_list = []
        for jerk in jerk_list:
            #final_jerk_value += np.linalg.norm(jerk)
            step_jerk_list.append(np.linalg.norm(jerk))
        return step_jerk_list

    def compute_heading_error_list(self, vehicle, lane):
        heading_error_list = []
        for i in range(1, self.config['seq_traj_len'] + 1):
            theta = vehicle.traj_wp_list[i]['yaw'] 
            long_now, lateral_now = lane.local_coordinates(vehicle.traj_wp_list[i]['position'])
            road_heading_theta = lane.heading_theta_at(long_now)
            theta_error = self.wrap_angle(theta - road_heading_theta)
            heading_error_list.append(np.abs(theta_error))
        return heading_error_list

    def compute_speed_list(self, vehicle):
        speed_list = []
        for i in range(1, self.config['seq_traj_len'] + 1):
            speed = vehicle.traj_wp_list[i]['speed']
            speed_list.append(speed)
        return speed_list

    def compute_avg_lateral_cum(self, vehicle, lane):
        # Compute lateral distance for each wp
        # average the factor by seq traj len
        # For example, if traj len is 10, then i = 1, 2, ... 10
        lateral_cum = 0
        for i in range(1, self.config['seq_traj_len'] + 1):
            long_now, lateral_now = lane.local_coordinates(vehicle.traj_wp_list[i]['position'])
            lateral_cum += np.abs(lateral_now)
        avg_lateral_cum = lateral_cum / float(self.config['seq_traj_len'])
        return avg_lateral_cum

    def switch_to_third_person_view(self) -> None:
        if self.main_camera is None:
            return
        self.main_camera.reset()
        if self.config["prefer_track_agent"] is not None and self.config["prefer_track_agent"] in self.vehicles.keys():
            new_v = self.vehicles[self.config["prefer_track_agent"]]
            current_track_vehicle = new_v
        else:
            if self.main_camera.is_bird_view_camera():
                current_track_vehicle = self.current_track_vehicle
            else:
                vehicles = list(self.engine.agents.values())
                if len(vehicles) <= 1:
                    return
                if self.current_track_vehicle in vehicles:
                    vehicles.remove(self.current_track_vehicle)
                new_v = get_np_random().choice(vehicles)
                current_track_vehicle = new_v
        self.main_camera.track(current_track_vehicle)
        return

    def switch_to_top_down_view(self):
        self.main_camera.stop_track()

    def _get_step_return(self, actions, engine_info):
        # update obs, dones, rewards, costs, calculate done at first !
        obses = {}
        done_infos = {}
        cost_infos = {}
        reward_infos = {}
        rewards = {}
        for v_id, v in self.vehicles.items():
            o = self.observations[v_id].observe(v)
            self.vel_speed = v.last_spd
            obses[v_id] =  o # o_dict
            if self.config["dense_reward"]:
                rewards[v_id], reward_infos[v_id] = self.reward_function_dense(v_id)
            else:
                rewards[v_id], reward_infos[v_id] = self.reward_function(v_id)
            done_function_result, done_infos[v_id] = self.done_function(v_id)
            _, cost_infos[v_id] = self.cost_function(v_id)
            done = done_function_result or self.dones[v_id]
            self.dones[v_id] = done

        should_done = engine_info.get(REPLAY_DONE, False
                                      ) or (self.config["horizon"] and self.episode_steps >= self.config["horizon"])
        termination_infos = self.for_each_vehicle(auto_termination, should_done)

        step_infos = concat_step_infos([
            engine_info,
            done_infos,
            reward_infos,
            cost_infos,
            termination_infos,
        ])

        if should_done:
            for k in self.dones:
                self.dones[k] = True

        dones = {k: self.dones[k] for k in self.vehicles.keys()}
        for v_id, r in rewards.items():
            self.episode_rewards[v_id] += r
            step_infos[v_id]["episode_reward"] = self.episode_rewards[v_id]
            self.episode_lengths[v_id] += 1
            step_infos[v_id]["episode_length"] = self.episode_lengths[v_id]
            step_infos[v_id]["envstep"] = self.env_step_num

        if not self.is_multi_agent:
            return self._wrap_as_single_agent(obses), self._wrap_as_single_agent(rewards), \
                   self._wrap_as_single_agent(dones), self._wrap_as_single_agent(step_infos)
        else:
            return obses, rewards, dones, step_infos

    def setup_engine(self):
        super(MetaDriveControlEnv, self).setup_engine()
        self.engine.accept("b", self.switch_to_top_down_view)
        self.engine.accept("q", self.switch_to_third_person_view)
        from asaprl.utils.env_utils.traffic_manager_utils import MacroTrafficManager
        from asaprl.utils.env_utils.map_manager_utils import MacroMapManager
        self.engine.register_manager("map_manager", MacroMapManager())
        self.engine.register_manager("traffic_manager", MacroTrafficManager())

    def _reset_global_seed(self, force_seed=None):
        current_seed = force_seed if force_seed is not None else \
            get_np_random(self._DEBUG_RANDOM_SEED).randint(self.start_seed, self.start_seed + self.env_num)
        self.seed(current_seed)

    def _preprocess_macro_actions(self, actions: Union[np.ndarray, Dict[AnyStr, np.ndarray]]) \
            -> Union[np.ndarray, Dict[AnyStr, np.ndarray]]:
        if not self.is_multi_agent:
            actions = int(actions)
            actions = {v_id: actions for v_id in self.vehicles.keys()}
        else:
            if self.config["vehicle_config"]["action_check"]:
                # Check whether some actions are not provided.
                given_keys = set(actions.keys())
                have_keys = set(self.vehicles.keys())
                assert given_keys == have_keys, "The input actions: {} have incompatible keys with existing {}!".format(
                    given_keys, have_keys
                )
            else:
                # That would be OK if extra actions is given. This is because, when evaluate a policy with naive
                # implementation, the "termination observation" will still be given in T=t-1. And at T=t, when you
                # collect action from policy(last_obs) without masking, then the action for "termination observation"
                # will still be computed. We just filter it out here.
                actions = {v_id: actions[v_id] for v_id in self.vehicles.keys()}
        return actions

    def _preprocess_macro_waypoints(self, waypoint_list: Union[np.ndarray, Dict[AnyStr, np.ndarray]]) \
            -> Union[np.ndarray, Dict[AnyStr, np.ndarray]]:
        if not self.is_multi_agent:
            actions = waypoint_list
            actions = {v_id: actions for v_id in self.vehicles.keys()}
        return actions

    def _step_macro_simulator(self, actions):
        #simulation_frequency = 30  # 60 80
        simulation_frequency = self.config['seq_traj_len']
        policy_frequency = 1
        frames = int(simulation_frequency / policy_frequency)
        self.time = 0
        wps = actions
        for frame in range(frames):
            # we use frame to update robot position, and use wps to represent the whole trajectory
            scene_manager_before_step_infos = self.engine.before_step_macro(frame, wps)
            self.engine.step()
            scene_manager_after_step_infos = self.engine.after_step()

            for agent in self.engine.agent_manager._active_objects.values():
                if agent.on_broken_line: self.last_on_broken_line += 1
        #scene_manager_after_step_infos = self.engine.after_step()
        return merge_dicts(
            scene_manager_after_step_infos, scene_manager_before_step_infos, allow_new_keys=True, without_copy=True
        ), simulation_frequency

    def _get_reset_return(self):
        ret = {}
        self.engine.after_step()
        o = None
        print('episode reward: {}'.format(self.episode_rwd))
        self.episode_rwd = 0
        self.step_num = 0
        self.env_step_num = 0
        for v_id, v in self.vehicles.items():
            self.observations[v_id].reset(self, v)
            ret[v_id] = self.observations[v_id].observe(v)
            o = self.observations[v_id].observe(v)
            v_o = np.ones([200, 200, 1]) * v.last_spd * 0.01
            o_dict = {}
            o_dict['birdview'] = o 
            o_dict['speed'] = v.last_spd
            self.vel_speed = 0

            if hasattr(v, 'macro_succ'):
                v.macro_succ = False
            if hasattr(v, 'macro_crash'):
                v.macro_crash = False
            v.penultimate_state = {}
            v.penultimate_state['position'] = np.array([0,0])
            v.penultimate_state['yaw'] = 0 
            v.penultimate_state['speed'] = 0
            v.traj_wp_list = [] 
            v.traj_wp_list.append(copy.deepcopy(v.penultimate_state))
            v.traj_wp_list.append(copy.deepcopy(v.penultimate_state))
            v.last_spd = 0

        self.already_go_dist = 0
        self._compute_navi_dist = True 
        self.navi_distance = 100.0
        self.remove_init_stop = True
        if self.remove_init_stop:
            return o #o_dict
        for i in range(8):
            o, r, d, info = self.step(self.action_type.actions_indexes["Holdon"])
        for v_id ,v in self.vehicles.items():
            if hasattr(v, 'macro_succ'):
                p = self.engine.get_policy(v.name)
                target_speed = p.NORMAL_SPEED * 0.1
                v.set_velocity(v.heading, target_speed)
        for i in range(1):
            o, r, d, info = self.step(self.action_type.actions_indexes["IDLE"])
            o = o
        return o

    def lazy_init(self):
        """
        Only init once in runtime, variable here exists till the close_env is called
        :return: None
        """
        # It is the true init() func to create the main vehicle and its module, to avoid incompatible with ray
        if engine_initialized():
            return
        self.engine = initialize_engine(self.config)
        # engine setup
        self.setup_engine()
        # other optional initialization
        self._after_lazy_init()

    def get_single_observation(self, _=None):
        o = TopDownMultiChannel(
            self.config["vehicle_config"],
            self.config["on_screen"],
            self.config["rgb_clip"],
            frame_stack=3,
            post_stack=10,
            frame_skip=1,
            resolution=(200, 200),
            max_distance=50
        )
        return o
    
    def wrap_angle(self, angle_in_rad):
        while (angle_in_rad > np.pi):
            angle_in_rad -= 2 * np.pi
        while (angle_in_rad <= -np.pi):
            angle_in_rad += 2 * np.pi
        return angle_in_rad


register(
    id='Control-v1',
    entry_point='asaprl.envs:MetaDriveControlEnv',
)
