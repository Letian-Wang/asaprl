from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.utils import Config, safe_clip_for_small_array
from asaprl.utils.env_utils.navigation_utils import SkillNodeNavigation
from typing import Union, Dict, AnyStr, Tuple
from asaprl.utils.env_utils.idm_policy_utils import MacroIDMPolicy
from metadrive.component.vehicle.vehicle_type import DefaultVehicle
import copy
import numpy as np

class MacroDefaultRuleExpertVehicle(DefaultVehicle):

    def __init__(self, vehicle_config: Union[dict, Config] = None, name: str = None, random_seed=None):
        super(MacroDefaultRuleExpertVehicle, self).__init__(vehicle_config, name, random_seed)
        self.macro_succ = False
        self.macro_crash = False
        self.last_spd = 0
        self.last_acc = 0
        self.last_macro_position = self.last_position
        self.v_wps = [[0,0], [1,1]]
        self.v_indx = 1
        self.physics_world_step_size = self.engine.global_config["physics_world_step_size"]
        self.penultimate_state = {}
        self.penultimate_state['position'] = np.array([0,0]) #self.last_position
        self.penultimate_state['yaw'] = 0 
        self.penultimate_state['speed'] = 0
        self.traj_wp_list = [] 
        self.traj_wp_list.append(copy.deepcopy(self.penultimate_state))
        self.traj_wp_list.append(copy.deepcopy(self.penultimate_state))

    def before_macro_step(self, macro_action):
        self.last_macro_position = self.position
        # if macro_action == 0:
        #     self.last_macro_position = self.position
        # else:
        #     pass
        # return
    def add_navigation(self):
        navi = SkillNodeNavigation
        self.navigation = \
            navi(self.engine,
                 show_navi_mark=self.engine.global_config["vehicle_config"]["show_navi_mark"],
                 random_navi_mark_color=self.engine.global_config["vehicle_config"]["random_navi_mark_color"],
                 show_dest_mark=self.engine.global_config["vehicle_config"]["show_dest_mark"],
                 show_line_to_dest=self.engine.global_config["vehicle_config"]["show_line_to_dest"],
                 seq_traj_len = self.engine.global_config["seq_traj_len"],
                 show_seq_traj = self.engine.global_config["show_seq_traj"])
    def _update_overtake_stat(self):
        if self.config["overtake_stat"] and self.lidar.available:
            # surrounding_vs = self.lidar.get_surrounding_vehicles()
            surrounding_obj = self.lidar.get_surrounding_objects(self)
            routing = self.navigation
            ckpt_idx = routing._target_checkpoints_index
            for surrounding_v in surrounding_obj:
                if surrounding_v.lane_index[:-1] == (routing.checkpoints[ckpt_idx[0]], routing.checkpoints[ckpt_idx[1]
                                                                                                           ]):
                    if self.lane.local_coordinates(self.position)[0] - \
                            self.lane.local_coordinates(surrounding_v.position)[0] < 0:
                        self.front_vehicles.add(surrounding_v)
                        if surrounding_v in self.back_vehicles:
                            self.back_vehicles.remove(surrounding_v)
                    else:
                        self.back_vehicles.add(surrounding_v)
        return {"overtake_vehicle_num": self.get_overtake_num()}

class MacroDefaultVehicle(DefaultVehicle):
    def __init__(self, vehicle_config: Union[dict, Config] = None, name: str = None, random_seed=None):
        super(MacroDefaultVehicle, self).__init__(vehicle_config, name, random_seed)
        self.macro_succ = False
        self.macro_crash = False
        self.last_spd = 0
        self.last_acc = 0
        self.last_macro_position = self.last_position
        self.v_wps = [[0,0], [1,1]]
        self.v_indx = 1
        self.physics_world_step_size = self.engine.global_config["physics_world_step_size"]
        self.penultimate_state = {}
        self.penultimate_state['position'] = np.array([0,0]) #self.last_position
        self.penultimate_state['yaw'] = 0 
        self.penultimate_state['speed'] = 0
        self.traj_wp_list = [] 
        self.traj_wp_list.append(copy.deepcopy(self.penultimate_state))
        self.traj_wp_list.append(copy.deepcopy(self.penultimate_state))

    def before_macro_step(self, macro_action):
        if macro_action == 0:
            self.last_macro_position = self.position
        else:
            pass
        return
    def add_navigation(self):
        navi = SkillNodeNavigation
        self.navigation = \
            navi(self.engine,
                 show_navi_mark=self.engine.global_config["vehicle_config"]["show_navi_mark"],
                 random_navi_mark_color=self.engine.global_config["vehicle_config"]["random_navi_mark_color"],
                 show_dest_mark=self.engine.global_config["vehicle_config"]["show_dest_mark"],
                 show_line_to_dest=self.engine.global_config["vehicle_config"]["show_line_to_dest"],
                 seq_traj_len = self.engine.global_config["seq_traj_len"],
                 show_seq_traj = self.engine.global_config["show_seq_traj"])
    def _update_overtake_stat(self):
        if self.config["overtake_stat"] and self.lidar.available:
            # surrounding_vs = self.lidar.get_surrounding_vehicles()
            surrounding_obj = self.lidar.get_surrounding_objects(self)
            routing = self.navigation
            ckpt_idx = routing._target_checkpoints_index
            for surrounding_v in surrounding_obj:
                if surrounding_v.lane_index[:-1] == (routing.checkpoints[ckpt_idx[0]], routing.checkpoints[ckpt_idx[1]
                                                                                                           ]):
                    if self.lane.local_coordinates(self.position)[0] - \
                            self.lane.local_coordinates(surrounding_v.position)[0] < 0:
                        self.front_vehicles.add(surrounding_v)
                        if surrounding_v in self.back_vehicles:
                            self.back_vehicles.remove(surrounding_v)
                    else:
                        self.back_vehicles.add(surrounding_v)
        return {"overtake_vehicle_num": self.get_overtake_num()}

class MacroBaseVehicle(BaseVehicle):

    def __init__(self, vehicle_config: Union[dict, Config] = None, name: str = None, random_seed=None):
        super(MacroBaseVehicle, self).__init__(vehicle_config, name, random_seed)
        self.macro_succ = False
        self.macro_crash = False
        self.replace_navigation()
    def add_navigation(self):
        navi = SkillNodeNavigation
        self.navigation = \
            navi(self.engine,
                 show_navi_mark=self.engine.global_config["vehicle_config"]["show_navi_mark"],
                 random_navi_mark_color=self.engine.global_config["vehicle_config"]["random_navi_mark_color"],
                 show_dest_mark=self.engine.global_config["vehicle_config"]["show_dest_mark"],
                 show_line_to_dest=self.engine.global_config["vehicle_config"]["show_line_to_dest"])


    def replace_navigation(self):
        navi = SkillNodeNavigation
        self.navigation = \
            navi(self.engine,
                 show_navi_mark=self.engine.global_config["vehicle_config"]["show_navi_mark"],
                 random_navi_mark_color=self.engine.global_config["vehicle_config"]["random_navi_mark_color"],
                 show_dest_mark=self.engine.global_config["vehicle_config"]["show_dest_mark"],
                 show_line_to_dest=self.engine.global_config["vehicle_config"]["show_line_to_dest"])
