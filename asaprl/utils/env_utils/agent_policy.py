import logging

import numpy as np

from metadrive.component.vehicle_module.PID_controller import PIDController
from metadrive.policy.base_policy import BasePolicy
from metadrive.policy.manual_control_policy import ManualControlPolicy
from metadrive.utils.math_utils import not_zero, wrap_to_pi, point_distance
from metadrive.utils.scene_utils import is_same_lane_index, is_following_lane_index
from metadrive.engine.core.manual_controller import KeyboardController, SteeringWheelController
from metadrive.utils import clip
from metadrive.examples import expert
from metadrive.policy.env_input_policy import EnvInputPolicy
from direct.controls.InputState import InputState
from metadrive.engine.engine_utils import get_global_config
from metadrive.utils import norm 


#from metadrive.policy.discrete_policy import ActionType, DiscreteMetaAction
class FrontBackObjects:

    def __init__(self, front_ret, back_ret, front_dist, back_dist):
        self.front_objs = front_ret
        self.back_objs = back_ret
        self.front_dist = front_dist
        self.back_dist = back_dist

    def left_lane_exist(self):
        return True if self.front_dist[0] is not None else False

    def right_lane_exist(self):
        return True if self.front_dist[-1] is not None else False

    def has_front_object(self):
        return True if self.front_objs[1] is not None else False

    def has_back_object(self):
        return True if self.back_objs[1] is not None else False

    def has_left_front_object(self):
        return True if self.front_objs[0] is not None else False

    def has_left_back_object(self):
        return True if self.back_objs[0] is not None else False

    def has_right_front_object(self):
        return True if self.front_objs[-1] is not None else False

    def has_right_back_object(self):
        return True if self.back_objs[-1] is not None else False

    def front_object(self):
        return self.front_objs[1]

    def left_front_object(self):
        return self.front_objs[0]

    def right_front_object(self):
        return self.front_objs[-1]

    def back_object(self):
        return self.back_objs[1]

    def left_back_object(self):
        return self.back_objs[0]

    def right_back_object(self):
        return self.back_objs[-1]

    def left_front_min_distance(self):
        assert self.left_lane_exist(), "left lane doesn't exist"
        return self.front_dist[0]

    def right_front_min_distance(self):
        assert self.right_lane_exist(), "right lane doesn't exist"
        return self.front_dist[-1]

    def front_min_distance(self):
        return self.front_dist[1]

    def left_back_min_distance(self):
        assert self.left_lane_exist(), "left lane doesn't exist"
        return self.back_dist[0]

    def right_back_min_distance(self):
        assert self.right_lane_exist(), "right lane doesn't exist"
        return self.back_dist[-1]

    def back_min_distance(self):
        return self.back_dist[1]

    @classmethod
    def get_find_front_back_objs(cls, objs, lane, position, max_distance, ref_lanes=None):
        """
        Find objects in front of/behind the lane and its left lanes/right lanes, return objs, dist.
        If ref_lanes is None, return filter results of this lane
        """
        if ref_lanes is not None:
            assert lane in ref_lanes
        idx = lane.index[-1]
        left_lane = ref_lanes[idx - 1] if idx > 0 and ref_lanes is not None else None
        right_lane = ref_lanes[idx + 1] if ref_lanes and idx + 1 < len(ref_lanes) is not None else None
        lanes = [left_lane, lane, right_lane]

        min_front_long = [max_distance if lane is not None else None for lane in lanes]
        min_back_long = [max_distance if lane is not None else None for lane in lanes]

        front_ret = [None, None, None]
        back_ret = [None, None, None]

        find_front_in_current_lane = [False, False, False]
        find_back_in_current_lane = [False, False, False]

        current_long = [lane.local_coordinates(position)[0] if lane is not None else None for lane in lanes]
        left_long = [lane.length - current_long[idx] if lane is not None else None for idx, lane in enumerate(lanes)]

        for i, lane in enumerate(lanes):
            if lane is None:
                continue
            for obj in objs:
                if obj.lane is lane:
                    long = lane.local_coordinates(obj.position)[0] - current_long[i]
                    if min_front_long[i] > long > 0:
                        min_front_long[i] = long
                        front_ret[i] = obj
                        find_front_in_current_lane[i] = True
                    if long < 0 and abs(long) < min_back_long[i]:
                        min_back_long[i] = abs(long)
                        back_ret[i] = obj
                        find_back_in_current_lane[i] = True

                elif not find_front_in_current_lane[i] and lane.is_previous_lane_of(obj.lane):
                    long = obj.lane.local_coordinates(obj.position)[0] + left_long[i]
                    if min_front_long[i] > long > 0:
                        min_front_long[i] = long
                        front_ret[i] = obj
                elif not find_back_in_current_lane[i] and obj.lane.is_previous_lane_of(lane):
                    long = obj.lane.length - obj.lane.local_coordinates(obj.position)[0] + current_long[i]
                    if min_back_long[i] > long:
                        min_back_long[i] = long
                        back_ret[i] = obj

        return cls(front_ret, back_ret, min_front_long, min_back_long)


class AgentSkillPolicy(BasePolicy):
    NORMAL_SPEED = 65  # 65
    ACC_FACTOR = 1.0

    def __init__(self, control_object, random_seed):
        super(AgentSkillPolicy, self).__init__(control_object=control_object, random_seed=random_seed)
        self.inputs = InputState()
        self.inputs.watchWithModifiers('accelerate', 'w')
        self.inputs.watchWithModifiers('deccelerate', 's')
        self.inputs.watchWithModifiers('laneLeft', 'a')
        self.inputs.watchWithModifiers('laneRight', 'd')
        self.manual = False

        self.heading_pid = PIDController(1.7, 0.01, 3.5)
        self.lateral_pid = PIDController(0.2, .002, 0.3)
        self.DELTA_SPEED = 10
        self.DELTA = 10
        self.target_speed = self.NORMAL_SPEED
        self.stop_label = True
        self.base_pos = self.control_object.position
        self.base_heading = self.control_object.heading_theta
        self.last_heading = self.base_heading

    def convert_wp_to_world_coord(self, rbt_pos, rbt_heading, wp, visual=False):
        compose_visual = 0
        if visual:
            compose_visual += 0 # 4.51 / 2
        theta = np.arctan2(wp[1], wp[0] + compose_visual)
        rbt_heading = rbt_heading # np.arctan2(rbt_heading[1], rbt_heading[0])
        theta = wrap_to_pi(rbt_heading) + wrap_to_pi(theta)
        norm_len = norm(wp[0] + compose_visual, wp[1])
        position = rbt_pos
        #position += 4.51 /2
        heading = np.sin(theta) * norm_len
        side = np.cos(theta) * norm_len
        return position[0] + side, position[1] + heading

    def convert_waypoint_list_coord(self, rbt_pos, rbt_heading, wp_list, visual = False):
        wp_w_list = []
        LENGTH = 4.51
        for wp in wp_list:
            wp_w = self.convert_wp_to_world_coord(rbt_pos, rbt_heading, wp, visual)
            wp_w_list.append(wp_w)
        return wp_w_list

    def act(self, *args, **kwargs):
        if (self.control_object.arrive_destination and hasattr(self.control_object, 'macro_succ')):
            self.control_object.macro_succ = True
        if (self.control_object.crash_vehicle and hasattr(self.control_object, 'crash_vehicle')):
            self.control_object.macro_crash = True

        if (len(args) >= 2):
            macro_action = args[1]
        frame = args[1]
        wp_list = args[2]
        ego_vehicle = self.control_object
        if frame ==0:
            self.base_pos = ego_vehicle.position
            self.base_heading = ego_vehicle.heading_theta
            self.control_object.v_wps = self.convert_waypoint_list_coord(self.base_pos, self.base_heading, wp_list, True)
            self.control_object.penultimate_state = self.control_object.traj_wp_list[-2] # if len(wp_list)>2 else self.control_object.traj_wp_list[-1]
            new_state = {}        
            new_state['position'] = ego_vehicle.position
            new_state['yaw'] = ego_vehicle.heading_theta
            new_state['speed'] = ego_vehicle.last_spd
            self.control_object.traj_wp_list = []
            self.control_object.traj_wp_list.append(new_state)
        self.control_object.v_indx = frame 
        wp_list = self.convert_waypoint_list_coord(self.base_pos, self.base_heading, wp_list)
        current_pos = np.array(wp_list[frame])
        target_pos = np.array(wp_list[frame+1])

        # directly set the position of the agent
        diff = target_pos - current_pos 
        norm = np.sqrt(diff[0] * diff[0] + diff[1] * diff[1])
        if abs(norm) < 0.001:
            heading_theta_at = self.last_heading
        else:
            direction = diff / norm 
            heading_theta_at = np.arctan2(direction[1], direction[0])
        ego_vehicle.set_position(target_pos)
        ego_vehicle.set_heading_theta(heading_theta_at)
        ego_vehicle.last_spd = norm / ego_vehicle.physics_world_step_size
        new_state = {}
        new_state['position'] = target_pos
        new_state['yaw'] = heading_theta_at
        new_state['speed'] = ego_vehicle.last_spd
        self.control_object.traj_wp_list.append(new_state)

        # these actions are actually not used
        self.last_heading = heading_theta_at 
        steering = 0 # self.steering_conrol_traj(lateral, heading_theta_at)
        throtle_brake = 0 #self.speed_control(target_vel)
        return [steering, throtle_brake]
        


    def get_neighboring_lanes(self):
        ref_lanes = self.control_object.navigation.current_ref_lanes
        lane = self.control_object.lane
        if ref_lanes is not None:
            assert lane in ref_lanes
        if self.after_end_of_lane(lane, self.control_object.position):
            if self.control_object.navigation.next_ref_lanes is not None:
                ref_lanes = self.control_object.navigation.next_ref_lanes
            for ref_lane in ref_lanes:
                if (self.target_lane.is_previous_lane_of(ref_lane)):
                    self.target_lane = ref_lane
        else:
            pass
        idx = lane.index[-1]
        left_lane = ref_lanes[idx - 1] if idx > 0 and ref_lanes is not None else None
        right_lane = ref_lanes[idx + 1] if idx + 1 < len(ref_lanes) and ref_lanes is not None else None
        lanes = [left_lane, lane, right_lane]
        return lanes

    def after_end_of_lane(self, lane, position):
        longitudinal, _ = lane.local_coordinates(position)
        return longitudinal > lane.length - lane.VEHICLE_LENGTH

    def follow_road(self) -> None:
        return

    def move_to_next_road(self):
        current_lanes = self.control_object.navigation.current_ref_lanes
        if self.routing_target_lane is None:
            self.routing_target_lane = self.control_object.lane
            return True if self.routing_target_lane in current_lanes else False
        if self.routing_target_lane not in current_lanes:
            for lane in current_lanes:
                if self.routing_target_lane.is_previous_lane_of(lane):
                    self.routing_target_lane = lane
                return True
            return False
        elif self.control_object.lane in current_lanes and self.routing_target_lane is not self.control_object.lane:
            self.routing_target_lane = self.control_object.lane
            return True
        else:
            return True

    def lane_change_policy(self):
        current_lanes = self.control_object.navigation.current_ref_lanes
        available_routing_index_range = [i for i in range(len(current_lanes))]
        next_lanes = self.control_object.navigation.next_ref_lanes
        lane_num_diff = len(current_lanes) - len(next_lanes) if next_lanes is not None else 0
        if lane_num_diff > 0:
            if current_lanes[0].is_previous_lane_of(next_lanes[0]):
                index_range = [i for i in range(len(next_lanes))]
            else:
                index_range = [i for i in range(lane_num_diff, len(current_lanes))]
            self.available_routing_index_range = index_range
            if self.routing_target_lane.index[-1] not in index_range:
                if self.routing_target_lane.index[-1] > index_range[-1]:
                    return current_lanes[self.routing_target_lane.index[-1] - 1]
                else:
                    return current_lanes[self.routing_target_lane.index[-1] - 1]

    def steering_conrol_traj(self, lat, lane_heading):
        # heading control following a lateral distance control
        ego_vehicle = self.control_object
        v_heading = ego_vehicle.heading_theta
        steering = self.heading_pid.get_result(wrap_to_pi(lane_heading - v_heading))
        steering += self.lateral_pid.get_result(-lat)
        return float(steering)

    def speed_control(self, target_speed):
        ego_vehicle = self.control_object
        ego_target_speed = not_zero(target_speed, 0.001)
        acceleration = self.ACC_FACTOR * (1 - np.power(max(ego_vehicle.speed, 0) / ego_target_speed, self.DELTA))
        return acceleration

