from metadrive.policy.idm_policy import IDMPolicy, FrontBackObjects
import logging

from pdb import set_trace

from cv2 import threshold
from metadrive.component.vehicle_module.PID_controller import PIDController
import math
import pdb
import numpy as np
from metadrive.utils.math_utils import not_zero, wrap_to_pi


class MacroIDMPolicy(IDMPolicy):

    def __init__(self, control_object, random_seed):
        super(MacroIDMPolicy, self).__init__(control_object=control_object, random_seed=random_seed)
        self.NORMAL_SPEED_CONST = 15
        self.NORMAL_SPEED = self.NORMAL_SPEED_CONST
        self.LANE_CHANGE_FREQ = 300

    def act(self, *args, **kwargs):
        # concat lane
        sucess = self.move_to_next_road()
        all_objects = self.control_object.lidar.get_surrounding_objects(self.control_object)
        self.set_target_speed()
        try:
            if sucess and self.enable_lane_change:
                # perform lane change due to routing
                acc_front_obj, acc_front_dist, steering_target_lane = self.lane_change_policy(all_objects)
            else:
                # can not find routing target lane
                surrounding_objects = FrontBackObjects.get_find_front_back_objs(
                    all_objects,
                    self.routing_target_lane,
                    self.control_object.position,
                    max_distance=self.MAX_LONG_DIST
                )
                acc_front_obj = surrounding_objects.front_object()
                acc_front_dist = surrounding_objects.front_min_distance()
                steering_target_lane = self.routing_target_lane
        except:
            # error fallback
            acc_front_obj = None
            acc_front_dist = 5
            steering_target_lane = self.routing_target_lane
            logging.warning("IDM bug! fall back")
            print("IDM bug! fall back")

        # control by PID and IDM
        steering = self.steering_control(steering_target_lane)
        acc = self.acceleration(acc_front_obj, acc_front_dist)
        return [steering, acc]

    def set_target_speed(self):
        # self.control_object.lane.index
        current_lane_index = self.control_object.lane_index[-1]
        speed_shift = self.engine.traffic_manager.speed_shift_lst[current_lane_index]
        self.NORMAL_SPEED = self.NORMAL_SPEED_CONST + speed_shift

class RuleExpertFrontBackObjects:
    def __init__(self, front_ret, back_ret, front_dist, back_dist):
        self.front_objs = front_ret
        self.back_objs = back_ret
        self.front_dist = front_dist
        self.back_dist = back_dist

    def left_left_lane_exist(self):
        return True if self.front_dist[0] is not None else False

    def left_lane_exist(self):
        return True if self.front_dist[1] is not None else False

    def right_lane_exist(self):
        return True if self.front_dist[-2] is not None else False

    def right_right_lane_exist(self):
        return True if self.front_dist[-1] is not None else False

    def has_front_object(self):
        return True if self.front_objs[2] is not None else False

    def has_back_object(self):
        return True if self.back_objs[2] is not None else False

    def has_left_left_front_object(self):
        return True if self.front_objs[0] is not None else False

    def has_left_left_back_object(self):
        return True if self.back_objs[0] is not None else False

    def has_left_front_object(self):
        return True if self.front_objs[1] is not None else False

    def has_left_back_object(self):
        return True if self.back_objs[1] is not None else False

    def has_right_front_object(self):
        return True if self.front_objs[-2] is not None else False

    def has_right_back_object(self):
        return True if self.back_objs[-2] is not None else False

    def has_right_right_front_object(self):
        return True if self.front_objs[-1] is not None else False

    def has_right_right_back_object(self):
        return True if self.back_objs[-1] is not None else False

    def front_object(self):
        return self.front_objs[2]

    def left_left_front_object(self):
        return self.front_objs[0]
 
    def left_front_object(self):
        return self.front_objs[1]

    def right_front_object(self):
        return self.front_objs[-2]

    def right_right_front_object(self):
        return self.front_objs[-1]

    def back_object(self):
        return self.back_objs[2]

    def left_left_back_object(self):
        return self.back_objs[0]

    def left_back_object(self):
        return self.back_objs[1]

    def right_back_object(self):
        return self.back_objs[-2]

    def right_right_back_object(self):
        return self.back_objs[-1]

    def left_left_front_min_distance(self):
        assert self.left_left_lane_exist(), "left left lane doesn't exist"
        return self.front_dist[0]

    def left_front_min_distance(self):
        assert self.left_lane_exist(), "left lane doesn't exist"
        return self.front_dist[1]

    def right_front_min_distance(self):
        assert self.right_lane_exist(), "right lane doesn't exist"
        return self.front_dist[-2]

    def right_right_front_min_distance(self):
        assert self.right_right_lane_exist(), "right right lane doesn't exist"
        return self.front_dist[-1]

    def front_min_distance(self):
        return self.front_dist[2]

    def left_left_back_min_distance(self):
        assert self.left_left_lane_exist(), "left left lane doesn't exist"
        return self.back_dist[0]

    def left_back_min_distance(self):
        assert self.left_lane_exist(), "left lane doesn't exist"
        return self.back_dist[1]

    def right_back_min_distance(self):
        assert self.right_lane_exist(), "right lane doesn't exist"
        return self.back_dist[-2]

    def right_right_back_min_distance(self):
        assert self.right_right_lane_exist(), "right right lane doesn't exist"
        return self.back_dist[-1]

    def back_min_distance(self):
        return self.back_dist[2]

    @classmethod
    def get_find_front_back_objs(cls, objs, lane, ego_object, max_distance, ref_lanes=None, target_speed=25, ego_lane_max_front_distance=27, debug=False):
        """
        Find objects in front of/behind the lane and its left lanes/right lanes, return objs, dist.
        If ref_lanes is None, return filter results of this lane
        """
        position = ego_object.position
        if ref_lanes is not None:
            assert lane in ref_lanes
        idx = lane.index[-1] if ref_lanes is not None else None

        left_lane = ref_lanes[idx - 1] if ref_lanes is not None and idx > 0 else None
        left_left_lane = ref_lanes[idx - 2] if ref_lanes is not None and idx > 1 else None
        right_lane = ref_lanes[idx + 1] if ref_lanes is not None and idx + 1 < len(ref_lanes) else None
        right_right_lane = ref_lanes[idx + 2] if ref_lanes is not None and idx + 2 < len(ref_lanes) else None
        lanes = [left_left_lane, left_lane, lane, right_lane, right_right_lane]
        next_lanes = ego_object.navigation.next_ref_lanes

        min_front_long = [max_distance if lane is not None else None for lane in lanes]
        min_back_long = [max_distance if lane is not None else None for lane in lanes]

        front_ret = [None, None, None, None, None]
        back_ret = [None, None, None, None, None]

        find_front_in_current_lane = [None, None, False, False, False]
        find_back_in_current_lane = [None, None, False, False, False]

        current_long = [lane.local_coordinates(position)[0] if lane is not None else None for lane in lanes]
        left_long = [lane.length - current_long[idx] if lane is not None else None for idx, lane in enumerate(lanes)]

        for i, lane in enumerate(lanes):
            if lane is None:
                continue
            for obj in objs:
                if obj.lane is lane:
                    long = lane.local_coordinates(obj.position)[0] - current_long[i]
                    if ego_lane_max_front_distance > long > 0 and min_front_long[i] > long:
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

                elif obj.lane not in lanes and next_lanes is not None and obj.lane not in next_lanes:
                    obj_abs_long = lane.local_coordinates(obj.position)[0]
                    ego_abs_long = lane.local_coordinates(position)[0]

                    obj_front_pos = obj.position + (obj.LENGTH / 2 * math.cos(obj.heading_theta), obj.LENGTH / 2 * math.sin(obj.heading_theta))
                    obj_back_pos = obj.position - (obj.LENGTH / 2 * math.cos(obj.heading_theta), obj.LENGTH / 2 * math.sin(obj.heading_theta))

                    obj_front_05_pos = obj_front_pos + (obj.speed * 0.5 * math.cos(obj.heading_theta), obj.speed * 0.5 * math.sin(obj.heading_theta))
                    obj_back_05_pos = obj_back_pos + (obj.speed * 0.5 * math.cos(obj.heading_theta), obj.speed * 0.5 * math.sin(obj.heading_theta))

                    obj_front_10_pos = obj_front_pos + (obj.speed * 1 * math.cos(obj.heading_theta), obj.speed * 1 * math.sin(obj.heading_theta))
                    obj_back_10_pos = obj_back_pos + (obj.speed * 1 * math.cos(obj.heading_theta), obj.speed * 1 * math.sin(obj.heading_theta))

                    obj_front_15_pos = obj_front_pos + (obj.speed * 1.5 * math.cos(obj.heading_theta), obj.speed * 1.5 * math.sin(obj.heading_theta))
                    obj_back_15_pos = obj_back_pos + (obj.speed * 1.5 * math.cos(obj.heading_theta), obj.speed * 1.5 * math.sin(obj.heading_theta))

                    obj_front_20_pos = obj_front_pos + (obj.speed * 2 * math.cos(obj.heading_theta), obj.speed * 2 * math.sin(obj.heading_theta))
                    obj_back_20_pos = obj_back_pos + (obj.speed * 2 * math.cos(obj.heading_theta), obj.speed * 2 * math.sin(obj.heading_theta))

                    time_to_conflict = (obj_abs_long - ego_abs_long) / (ego_object.speed+0.001) if obj_abs_long - ego_abs_long > 0 else 0
                    obj_front_confict_pos = obj_front_pos + (obj.speed * time_to_conflict * math.cos(obj.heading_theta), obj.speed * time_to_conflict * math.sin(obj.heading_theta))
                    obj_back_confict_pos = obj_back_pos + (obj.speed * time_to_conflict * math.cos(obj.heading_theta), obj.speed * time_to_conflict * math.sin(obj.heading_theta))


                    obj_front_lat = lane.local_coordinates(obj_front_pos)[1]
                    obj_back_lat = lane.local_coordinates(obj_back_pos)[1]
                    obj_front_05_lat = lane.local_coordinates(obj_front_05_pos)[1]
                    obj_back_05_lat = lane.local_coordinates(obj_back_05_pos)[1]
                    obj_front_10_lat = lane.local_coordinates(obj_front_10_pos)[1]
                    obj_back_10_lat = lane.local_coordinates(obj_back_10_pos)[1]
                    obj_front_15_lat = lane.local_coordinates(obj_front_15_pos)[1]
                    obj_back_15_lat = lane.local_coordinates(obj_back_15_pos)[1]
                    obj_front_20_lat = lane.local_coordinates(obj_front_20_pos)[1]
                    obj_back_20_lat = lane.local_coordinates(obj_back_20_pos)[1]
                    obj_front_conflict_lat = lane.local_coordinates(obj_front_confict_pos)[1]
                    obj_back_conflict_lat = lane.local_coordinates(obj_back_confict_pos)[1]


                    if (obj_front_lat < lane.width/2 and obj_front_lat > - lane.width/2) \
                        or (obj_back_lat < lane.width/2 and obj_back_lat > - lane.width/2) \
                        or (obj_front_05_lat < lane.width/2 and obj_front_05_lat > - lane.width/2) \
                        or (obj_back_05_lat < lane.width/2 and obj_back_05_lat > - lane.width/2) \
                        or (obj_front_10_lat < lane.width/2 and obj_front_10_lat > - lane.width/2) \
                        or (obj_back_10_lat < lane.width/2 and obj_back_10_lat > - lane.width/2) \
                        and (obj_abs_long >= 0 and obj_abs_long >= ego_abs_long and obj_abs_long <= lane.length+10):
                        # or (obj_front_15_lat < lane.width/2 and obj_front_15_lat > - lane.width/2) \
                        # or (obj_back_15_lat < lane.width/2 and obj_back_15_lat > - lane.width/2) \
                        # or (obj_front_conflict_lat < lane.width/2 and obj_front_conflict_lat > - lane.width/2) \
                        # or (obj_back_conflict_lat < lane.width/2 and obj_back_conflict_lat > - lane.width/2):
                        # or ((obj_front_20_lat < lane.width/2 and obj_front_20_lat > - lane.width/2) and (obj_abs_long >= 0 and obj_abs_long <= lane.length)) \
                        # or ((obj_back_20_lat < lane.width/2 and obj_back_20_lat > - lane.width/2) and (obj_abs_long >= 0 and obj_abs_long <= lane.length)):

                        long = lane.local_coordinates(obj.position)[0] - current_long[i]

                        if ego_lane_max_front_distance > long > 0 and min_front_long[i] > long:
                            min_front_long[i] = long
                            front_ret[i] = obj
                            find_front_in_current_lane[i] = True
                        if long < 0 and abs(long) < min_back_long[i]:
                            min_back_long[i] = abs(long)
                            back_ret[i] = obj
                            find_back_in_current_lane[i] = True

        return cls(front_ret, back_ret, min_front_long, min_back_long)

class RuleExpertIDMPolicyHighway(IDMPolicy):

    def __init__(self, control_object, random_seed):
        print('RuleExpertIDMPolicyHighway1')
        super(RuleExpertIDMPolicyHighway, self).__init__(control_object=control_object, random_seed=random_seed)
        print('RuleExpertIDMPolicyHighway2')
        self.NORMAL_SPEED_CONST = 25
        self.NORMAL_SPEED = self.NORMAL_SPEED_CONST
        self.LANE_CHANGE_FREQ = 150
        self.LANE_CHANGE_SPEED_INCREASE = 2
        self.DISTANCE_WANTED = 18
        self.TIME_WANTED = 0
        self.TIME_WANTED_EMERGENCY = 0.05
        self.lateral_pid = PIDController(0.2, .002, 0.1)
        self.pdb_counter = 0
        self.has_vehicle_on_conflict_lane = None

    def lane_change_policy(self, all_objects):
        current_lanes = self.control_object.navigation.current_ref_lanes
        if self.pdb_counter % 10 == 0:
            surrounding_objects = RuleExpertFrontBackObjects.get_find_front_back_objs(all_objects, self.routing_target_lane, self.control_object, self.MAX_LONG_DIST, current_lanes, self.NORMAL_SPEED, debug=True)
        else:
            surrounding_objects = RuleExpertFrontBackObjects.get_find_front_back_objs(
                all_objects, self.routing_target_lane, self.control_object, self.MAX_LONG_DIST, current_lanes, self.NORMAL_SPEED
            )
        self.available_routing_index_range = [i for i in range(len(current_lanes))]
        next_lanes = self.control_object.navigation.next_ref_lanes
        lane_num_diff = len(current_lanes) - len(next_lanes) if next_lanes is not None else 0

        # We have to perform lane changing because the number of lanes in next road is less than current road
        if lane_num_diff > 0:
            print("lane num decreasing happened in left road or right road")
            # lane num decreasing happened in left road or right road
            if current_lanes[0].is_previous_lane_of(next_lanes[0]):
                index_range = [i for i in range(len(next_lanes))]
            else:
                index_range = [i for i in range(lane_num_diff, len(current_lanes))]
            self.available_routing_index_range = index_range
            if self.routing_target_lane.index[-1] not in index_range:
                # not on suitable lane do lane change !!!
                if self.routing_target_lane.index[-1] > index_range[-1]:
                    # change to left
                    if surrounding_objects.left_back_min_distance(
                    ) < self.SAFE_LANE_CHANGE_DISTANCE or surrounding_objects.left_front_min_distance() < 5:
                        # creep to wait
                        self.target_speed = self.CREEP_SPEED
                        return surrounding_objects.front_object(), surrounding_objects.front_min_distance(
                        ), self.routing_target_lane
                    else:
                        # it is time to change lane!
                        self.target_speed = self.NORMAL_SPEED
                        return surrounding_objects.left_front_object(), surrounding_objects.left_front_min_distance(), \
                               current_lanes[self.routing_target_lane.index[-1] - 1]
                else:
                    # change to right
                    if surrounding_objects.right_back_min_distance(
                    ) < self.SAFE_LANE_CHANGE_DISTANCE or surrounding_objects.right_front_min_distance() < 5:
                        # unsafe, creep and wait
                        self.target_speed = self.CREEP_SPEED
                        return surrounding_objects.front_object(), surrounding_objects.front_min_distance(
                        ), self.routing_target_lane,
                    else:
                        # change lane
                        self.target_speed = self.NORMAL_SPEED
                        return surrounding_objects.right_front_object(), surrounding_objects.right_front_min_distance(), \
                               current_lanes[self.routing_target_lane.index[-1] + 1]

        # print surrounding object speed
        # print(44.0 if not surrounding_objects.left_left_lane_exist() else "{:2.1f}".format(surrounding_objects.left_left_front_min_distance()) if surrounding_objects.left_left_front_min_distance() is not None else 44.0, 44.0 if not surrounding_objects.left_lane_exist() else "{:2.1f}".format(surrounding_objects.left_front_min_distance()) if surrounding_objects.left_front_min_distance() is not None else 44.0, "{:2.1f}".format(surrounding_objects.front_min_distance()) if surrounding_objects.front_min_distance() is not None else 44.0, 44.0 if not surrounding_objects.right_lane_exist() else "{:2.1f}".format(surrounding_objects.right_front_min_distance()) if surrounding_objects.right_front_min_distance() is not None else 44.0, 44.0 if not surrounding_objects.right_right_lane_exist() else "{:2.1f}".format(surrounding_objects.right_right_front_min_distance()) if surrounding_objects.right_right_front_min_distance() is not None else 44.0)
        # print(44.0 if not surrounding_objects.left_left_lane_exist() else "{:2.1f}".format(surrounding_objects.left_left_back_min_distance()) if surrounding_objects.left_left_back_min_distance() is not None else 44.0, 44.0 if not surrounding_objects.left_lane_exist() else "{:2.1f}".format(surrounding_objects.left_back_min_distance()) if surrounding_objects.left_back_min_distance() is not None else 44.0, "{:2.1f}".format(surrounding_objects.back_min_distance()) if surrounding_objects.back_min_distance() is not None else 44.0, 44.0 if not surrounding_objects.right_lane_exist() else "{:2.1f}".format(surrounding_objects.right_back_min_distance()) if surrounding_objects.right_back_min_distance() is not None else 44.0, 44.0 if not surrounding_objects.right_right_lane_exist() else "{:2.1f}".format(surrounding_objects.right_right_back_min_distance()) if surrounding_objects.right_right_back_min_distance() is not None else 44.0)
        # print()
        # print(44.0 if not surrounding_objects.left_lane_exist() else "{:2.1f}".format(surrounding_objects.left_front_min_distance()) if surrounding_objects.left_front_min_distance() is not None else 44.0, "{:2.1f}".format(surrounding_objects.front_min_distance()) if surrounding_objects.front_min_distance() is not None else 44.0, 44.0 if not surrounding_objects.right_lane_exist() else "{:2.1f}".format(surrounding_objects.right_front_min_distance()) if surrounding_objects.right_front_min_distance() is not None else 44.0)
        # print(44.0 if not surrounding_objects.left_lane_exist() else "{:2.1f}".format(surrounding_objects.left_back_min_distance()) if surrounding_objects.left_back_min_distance() is not None else 44.0, "{:2.1f}".format(surrounding_objects.back_min_distance()) if surrounding_objects.back_min_distance() is not None else 44.0, 44.0 if not surrounding_objects.right_lane_exist() else "{:2.1f}".format(surrounding_objects.right_back_min_distance()) if surrounding_objects.right_back_min_distance() is not None else 44.0)

        # lane follow or active change lane/overtake for high driving speed
        if abs(self.control_object.speed - self.NORMAL_SPEED) > 3 and surrounding_objects.has_front_object(
        ) and abs(surrounding_objects.front_object().speed -
                  self.NORMAL_SPEED) > 3 and self.overtake_timer > self.LANE_CHANGE_FREQ:
            # may lane change
            left_left_front_speed = surrounding_objects.left_left_front_object().speed if surrounding_objects.has_left_left_front_object() else self.MAX_SPEED \
                if surrounding_objects.left_left_lane_exist() else None
            left_front_speed = surrounding_objects.left_front_object().speed if surrounding_objects.has_left_front_object() else self.MAX_SPEED \
                if surrounding_objects.left_lane_exist() else None
            front_speed = surrounding_objects.front_object().speed if surrounding_objects.has_front_object() else self.MAX_SPEED
            right_front_speed = surrounding_objects.right_front_object().speed if surrounding_objects.has_right_front_object() else self.MAX_SPEED \
                if surrounding_objects.right_lane_exist() else None
            right_right_front_speed = surrounding_objects.right_right_front_object().speed if surrounding_objects.has_right_right_front_object() else self.MAX_SPEED \
                if surrounding_objects.right_right_lane_exist() else None

            # nearby lane change
            if left_front_speed is not None and (right_front_speed is None or (right_front_speed is not None and left_front_speed >= right_front_speed)):
                # consider left front first
                if left_front_speed is not None and left_front_speed - front_speed > self.LANE_CHANGE_SPEED_INCREASE:
                    if (surrounding_objects.left_back_min_distance() > 12 and surrounding_objects.left_front_min_distance() > 3):
                        expect_lane_idx = current_lanes.index(self.routing_target_lane) - 1
                        if expect_lane_idx in self.available_routing_index_range:
                            print("lange to left")
                            return surrounding_objects.left_front_object(), surrounding_objects.left_front_min_distance(), \
                                current_lanes[expect_lane_idx]
                if right_front_speed is not None and right_front_speed - front_speed > self.LANE_CHANGE_SPEED_INCREASE:
                    if (surrounding_objects.right_back_min_distance() > 12 and surrounding_objects.right_front_min_distance() > 3):
                        expect_lane_idx = current_lanes.index(self.routing_target_lane) + 1
                        if expect_lane_idx in self.available_routing_index_range:
                            print("lange to right")
                            return surrounding_objects.right_front_object(), surrounding_objects.right_front_min_distance(), \
                                current_lanes[expect_lane_idx]
            else:
                # consider right front first
                if right_front_speed is not None and right_front_speed - front_speed > self.LANE_CHANGE_SPEED_INCREASE:
                    if (surrounding_objects.right_back_min_distance() > 12 and surrounding_objects.right_front_min_distance() > 3):
                        expect_lane_idx = current_lanes.index(self.routing_target_lane) + 1
                        if expect_lane_idx in self.available_routing_index_range:
                            print("lange to right")
                            return surrounding_objects.right_front_object(), surrounding_objects.right_front_min_distance(), \
                                current_lanes[expect_lane_idx]
                if left_front_speed is not None and left_front_speed - front_speed > self.LANE_CHANGE_SPEED_INCREASE:
                    if (surrounding_objects.left_back_min_distance() > 12 and surrounding_objects.left_front_min_distance() > 3):
                        expect_lane_idx = current_lanes.index(self.routing_target_lane) - 1
                        if expect_lane_idx in self.available_routing_index_range:
                            print("lange to left")
                            return surrounding_objects.left_front_object(), surrounding_objects.left_front_min_distance(), \
                                current_lanes[expect_lane_idx]

            # nearby nearby lane change
            if left_left_front_speed is not None and (right_right_front_speed is None or (right_right_front_speed is not None and left_left_front_speed >= right_right_front_speed)):
                if left_left_front_speed is not None and left_left_front_speed - front_speed > self.LANE_CHANGE_SPEED_INCREASE:
                    if (surrounding_objects.left_left_back_min_distance() > 12 and surrounding_objects.left_left_front_min_distance() > 3) \
                        and (surrounding_objects.left_back_min_distance() > 12 and surrounding_objects.left_front_min_distance() > 3):
                        expect_lane_idx = current_lanes.index(self.routing_target_lane) - 1
                        if expect_lane_idx in self.available_routing_index_range:
                            return surrounding_objects.left_front_object(), surrounding_objects.left_front_min_distance(), \
                                current_lanes[expect_lane_idx]
                if right_right_front_speed is not None and right_right_front_speed - front_speed > self.LANE_CHANGE_SPEED_INCREASE:
                    if (surrounding_objects.right_right_back_min_distance() > 12 and surrounding_objects.right_right_front_min_distance() > 3) \
                        and (surrounding_objects.right_back_min_distance() > 12 and surrounding_objects.right_front_min_distance() > 3):
                        expect_lane_idx = current_lanes.index(self.routing_target_lane) + 1
                        if expect_lane_idx in self.available_routing_index_range:
                            return surrounding_objects.right_front_object(), surrounding_objects.right_front_min_distance(), \
                                current_lanes[expect_lane_idx]
            else:
                if right_right_front_speed is not None and right_right_front_speed - front_speed > self.LANE_CHANGE_SPEED_INCREASE:
                    if (surrounding_objects.right_right_back_min_distance() > 12 and surrounding_objects.right_right_front_min_distance() > 3) \
                        and (surrounding_objects.right_back_min_distance() > 12 and surrounding_objects.right_front_min_distance() > 3):
                        expect_lane_idx = current_lanes.index(self.routing_target_lane) + 1
                        if expect_lane_idx in self.available_routing_index_range:
                            return surrounding_objects.right_front_object(), surrounding_objects.right_front_min_distance(), \
                                current_lanes[expect_lane_idx]
                if left_left_front_speed is not None and left_left_front_speed - front_speed > self.LANE_CHANGE_SPEED_INCREASE:
                    if (surrounding_objects.left_left_back_min_distance() > 12 and surrounding_objects.left_left_front_min_distance() > 3) \
                        and (surrounding_objects.left_back_min_distance() > 12 and surrounding_objects.left_front_min_distance() > 3):
                        expect_lane_idx = current_lanes.index(self.routing_target_lane) - 1
                        if expect_lane_idx in self.available_routing_index_range:
                            return surrounding_objects.left_front_object(), surrounding_objects.left_front_min_distance(), \
                                current_lanes[expect_lane_idx]

        # fall back to lane follow
        self.target_speed = self.NORMAL_SPEED
        self.overtake_timer += 1
        return surrounding_objects.front_object(), surrounding_objects.front_min_distance(), self.routing_target_lane

    def act(self, *args, **kwargs):
        self.pdb_counter += 1
        # concat lane
        sucess = self.move_to_next_road()
        all_objects = self.control_object.lidar.get_surrounding_objects(self.control_object)

        try:
            if sucess and self.enable_lane_change:
                # perform lane change due to routing
                # self.set_target_speed(all_objects)
                acc_front_obj, acc_front_dist, steering_target_lane = self.lane_change_policy(all_objects)
            else:
                # can not find routing target lane
                surrounding_objects = RuleExpertFrontBackObjects.get_find_front_back_objs(
                    all_objects,
                    self.routing_target_lane,
                    self.control_object,
                    max_distance=self.MAX_LONG_DIST,
                    target_speed=self.NORMAL_SPEED
                )
                acc_front_obj = surrounding_objects.front_object()
                acc_front_dist = surrounding_objects.front_min_distance()
                steering_target_lane = self.routing_target_lane
        except Exception as e:
            # error fallback
            surrounding_objects = RuleExpertFrontBackObjects.get_find_front_back_objs(
                all_objects,
                self.routing_target_lane,
                self.control_object,
                max_distance=self.MAX_LONG_DIST,
                target_speed = self.NORMAL_SPEED
            )
            # surrounding_objects = RuleExpertFrontBackObjects.get_find_front_back_objs(all_objects, self.routing_target_lane, self.control_object, max_distance=self.MAX_LONG_DIST, target_speed = self.NORMAL_SPEED)
            acc_front_obj = surrounding_objects.front_object()
            acc_front_dist = surrounding_objects.front_min_distance()
            steering_target_lane = self.routing_target_lane
            logging.warning("IDM bug! fall back")
            print("IDM bug! fall back")
            print('except log {}'.format(e))

        # control by PID and IDM
        steering = self.steering_control(steering_target_lane)
        acc = self.acceleration(acc_front_obj, acc_front_dist)

        return [steering, acc]

class RuleExpertIDMPolicyIntersection(IDMPolicy):

    def __init__(self, control_object, random_seed):
        super(RuleExpertIDMPolicyIntersection, self).__init__(control_object=control_object, random_seed=random_seed)
        self.NORMAL_SPEED_CONST = 25
        self.NORMAL_SPEED = self.NORMAL_SPEED_CONST
        self.LANE_CHANGE_FREQ = 150
        self.LANE_CHANGE_SPEED_INCREASE = 2
        self.DISTANCE_WANTED = 20
        self.TIME_WANTED = 0
        self.TIME_WANTED_EMERGENCY = 0.05
        self.lateral_pid = PIDController(0.2, .002, 0.1)
        self.pdb_counter = 0
        self.has_vehicle_on_conflict_lane = None

    def lane_change_policy_noback(self, all_objects):
        current_lanes = self.control_object.navigation.current_ref_lanes
        if self.pdb_counter % 10 == 0:
            # import pdb; pdb.set_trace()
            surrounding_objects = RuleExpertFrontBackObjects.get_find_front_back_objs(all_objects, self.routing_target_lane, self.control_object, self.MAX_LONG_DIST, current_lanes, self.NORMAL_SPEED, debug=True)
        else:
            surrounding_objects = RuleExpertFrontBackObjects.get_find_front_back_objs(
                all_objects, self.routing_target_lane, self.control_object, self.MAX_LONG_DIST, current_lanes, self.NORMAL_SPEED
            )
        self.available_routing_index_range = [i for i in range(len(current_lanes))]
        next_lanes = self.control_object.navigation.next_ref_lanes
        lane_num_diff = len(current_lanes) - len(next_lanes) if next_lanes is not None else 0

        # We have to perform lane changing because the number of lanes in next road is less than current road
        if lane_num_diff > 0:
            print("lane num decreasing happened in left road or right road")
            # lane num decreasing happened in left road or right road
            if current_lanes[0].is_previous_lane_of(next_lanes[0]):
                index_range = [i for i in range(len(next_lanes))]
            else:
                index_range = [i for i in range(lane_num_diff, len(current_lanes))]
            self.available_routing_index_range = index_range
            if self.routing_target_lane.index[-1] not in index_range:
                # not on suitable lane do lane change !!!
                if self.routing_target_lane.index[-1] > index_range[-1]:
                    # change to left
                    if surrounding_objects.left_back_min_distance(
                    ) < self.SAFE_LANE_CHANGE_DISTANCE or surrounding_objects.left_front_min_distance() < 5:
                        # creep to wait
                        self.target_speed = self.CREEP_SPEED
                        return surrounding_objects.front_object(), surrounding_objects.front_min_distance(
                        ), self.routing_target_lane
                    else:
                        # it is time to change lane!
                        self.target_speed = self.NORMAL_SPEED
                        return surrounding_objects.left_front_object(), surrounding_objects.left_front_min_distance(), \
                               current_lanes[self.routing_target_lane.index[-1] - 1]
                else:
                    # change to right
                    if surrounding_objects.right_back_min_distance(
                    ) < self.SAFE_LANE_CHANGE_DISTANCE or surrounding_objects.right_front_min_distance() < 5:
                        # unsafe, creep and wait
                        self.target_speed = self.CREEP_SPEED
                        return surrounding_objects.front_object(), surrounding_objects.front_min_distance(
                        ), self.routing_target_lane,
                    else:
                        # change lane
                        self.target_speed = self.NORMAL_SPEED
                        return surrounding_objects.right_front_object(), surrounding_objects.right_front_min_distance(), \
                               current_lanes[self.routing_target_lane.index[-1] + 1]

        # print surrounding object speed
        # print(44.0 if not surrounding_objects.left_left_lane_exist() else "{:2.1f}".format(surrounding_objects.left_left_front_min_distance()) if surrounding_objects.left_left_front_min_distance() is not None else 44.0, 44.0 if not surrounding_objects.left_lane_exist() else "{:2.1f}".format(surrounding_objects.left_front_min_distance()) if surrounding_objects.left_front_min_distance() is not None else 44.0, "{:2.1f}".format(surrounding_objects.front_min_distance()) if surrounding_objects.front_min_distance() is not None else 44.0, 44.0 if not surrounding_objects.right_lane_exist() else "{:2.1f}".format(surrounding_objects.right_front_min_distance()) if surrounding_objects.right_front_min_distance() is not None else 44.0, 44.0 if not surrounding_objects.right_right_lane_exist() else "{:2.1f}".format(surrounding_objects.right_right_front_min_distance()) if surrounding_objects.right_right_front_min_distance() is not None else 44.0)
        # print(44.0 if not surrounding_objects.left_left_lane_exist() else "{:2.1f}".format(surrounding_objects.left_left_back_min_distance()) if surrounding_objects.left_left_back_min_distance() is not None else 44.0, 44.0 if not surrounding_objects.left_lane_exist() else "{:2.1f}".format(surrounding_objects.left_back_min_distance()) if surrounding_objects.left_back_min_distance() is not None else 44.0, "{:2.1f}".format(surrounding_objects.back_min_distance()) if surrounding_objects.back_min_distance() is not None else 44.0, 44.0 if not surrounding_objects.right_lane_exist() else "{:2.1f}".format(surrounding_objects.right_back_min_distance()) if surrounding_objects.right_back_min_distance() is not None else 44.0, 44.0 if not surrounding_objects.right_right_lane_exist() else "{:2.1f}".format(surrounding_objects.right_right_back_min_distance()) if surrounding_objects.right_right_back_min_distance() is not None else 44.0)
        # print()
        # print(44.0 if not surrounding_objects.left_lane_exist() else "{:2.1f}".format(surrounding_objects.left_front_min_distance()) if surrounding_objects.left_front_min_distance() is not None else 44.0, "{:2.1f}".format(surrounding_objects.front_min_distance()) if surrounding_objects.front_min_distance() is not None else 44.0, 44.0 if not surrounding_objects.right_lane_exist() else "{:2.1f}".format(surrounding_objects.right_front_min_distance()) if surrounding_objects.right_front_min_distance() is not None else 44.0)
        # print(44.0 if not surrounding_objects.left_lane_exist() else "{:2.1f}".format(surrounding_objects.left_back_min_distance()) if surrounding_objects.left_back_min_distance() is not None else 44.0, "{:2.1f}".format(surrounding_objects.back_min_distance()) if surrounding_objects.back_min_distance() is not None else 44.0, 44.0 if not surrounding_objects.right_lane_exist() else "{:2.1f}".format(surrounding_objects.right_back_min_distance()) if surrounding_objects.right_back_min_distance() is not None else 44.0)

        # lane follow or active change lane/overtake for high driving speed
        if abs(self.control_object.speed - self.NORMAL_SPEED) > 3 and surrounding_objects.has_front_object(
        ) and abs(surrounding_objects.front_object().speed -
                  self.NORMAL_SPEED) > 3 and self.overtake_timer > self.LANE_CHANGE_FREQ:
            # may lane change
            left_left_front_speed = surrounding_objects.left_left_front_object().speed if surrounding_objects.has_left_left_front_object() else self.MAX_SPEED \
                if surrounding_objects.left_left_lane_exist() else None
            left_front_speed = surrounding_objects.left_front_object().speed if surrounding_objects.has_left_front_object() else self.MAX_SPEED \
                if surrounding_objects.left_lane_exist() else None
            front_speed = surrounding_objects.front_object().speed if surrounding_objects.has_front_object() else self.MAX_SPEED
            right_front_speed = surrounding_objects.right_front_object().speed if surrounding_objects.has_right_front_object() else self.MAX_SPEED \
                if surrounding_objects.right_lane_exist() else None
            right_right_front_speed = surrounding_objects.right_right_front_object().speed if surrounding_objects.has_right_right_front_object() else self.MAX_SPEED \
                if surrounding_objects.right_right_lane_exist() else None

            # nearby lane change
            if left_front_speed is not None and (right_front_speed is None or (right_front_speed is not None and left_front_speed >= right_front_speed)):
                # consider left front first
                if left_front_speed is not None and left_front_speed - front_speed > self.LANE_CHANGE_SPEED_INCREASE:
                    if (surrounding_objects.left_back_min_distance() > 12 and surrounding_objects.left_front_min_distance() > 3):
                        expect_lane_idx = current_lanes.index(self.routing_target_lane) - 1
                        if expect_lane_idx in self.available_routing_index_range:
                            print("lange to left")
                            return surrounding_objects.left_front_object(), surrounding_objects.left_front_min_distance(), \
                                current_lanes[expect_lane_idx]
                if right_front_speed is not None and right_front_speed - front_speed > self.LANE_CHANGE_SPEED_INCREASE:
                    if (surrounding_objects.right_back_min_distance() > 12 and surrounding_objects.right_front_min_distance() > 3):
                        expect_lane_idx = current_lanes.index(self.routing_target_lane) + 1
                        if expect_lane_idx in self.available_routing_index_range:
                            print("lange to right")
                            return surrounding_objects.right_front_object(), surrounding_objects.right_front_min_distance(), \
                                current_lanes[expect_lane_idx]
            else:
                # consider right front first
                if right_front_speed is not None and right_front_speed - front_speed > self.LANE_CHANGE_SPEED_INCREASE:
                    if (surrounding_objects.right_back_min_distance() > 12 and surrounding_objects.right_front_min_distance() > 3):
                        expect_lane_idx = current_lanes.index(self.routing_target_lane) + 1
                        if expect_lane_idx in self.available_routing_index_range:
                            print("lange to right")
                            return surrounding_objects.right_front_object(), surrounding_objects.right_front_min_distance(), \
                                current_lanes[expect_lane_idx]
                if left_front_speed is not None and left_front_speed - front_speed > self.LANE_CHANGE_SPEED_INCREASE:
                    if (surrounding_objects.left_back_min_distance() > 12 and surrounding_objects.left_front_min_distance() > 3):
                        expect_lane_idx = current_lanes.index(self.routing_target_lane) - 1
                        if expect_lane_idx in self.available_routing_index_range:
                            print("lange to left")
                            return surrounding_objects.left_front_object(), surrounding_objects.left_front_min_distance(), \
                                current_lanes[expect_lane_idx]

            # nearby nearby lane change
            if left_left_front_speed is not None and (right_right_front_speed is None or (right_right_front_speed is not None and left_left_front_speed >= right_right_front_speed)):
                if left_left_front_speed is not None and left_left_front_speed - front_speed > self.LANE_CHANGE_SPEED_INCREASE:
                    if (surrounding_objects.left_left_back_min_distance() > 12 and surrounding_objects.left_left_front_min_distance() > 3) \
                        and (surrounding_objects.left_back_min_distance() > 12 and surrounding_objects.left_front_min_distance() > 3):
                        expect_lane_idx = current_lanes.index(self.routing_target_lane) - 1
                        if expect_lane_idx in self.available_routing_index_range:
                            return surrounding_objects.left_front_object(), surrounding_objects.left_front_min_distance(), \
                                current_lanes[expect_lane_idx]
                if right_right_front_speed is not None and right_right_front_speed - front_speed > self.LANE_CHANGE_SPEED_INCREASE:
                    if (surrounding_objects.right_right_back_min_distance() > 12 and surrounding_objects.right_right_front_min_distance() > 3) \
                        and (surrounding_objects.right_back_min_distance() > 12 and surrounding_objects.right_front_min_distance() > 3):
                        expect_lane_idx = current_lanes.index(self.routing_target_lane) + 1
                        if expect_lane_idx in self.available_routing_index_range:
                            return surrounding_objects.right_front_object(), surrounding_objects.right_front_min_distance(), \
                                current_lanes[expect_lane_idx]
            else:
                if right_right_front_speed is not None and right_right_front_speed - front_speed > self.LANE_CHANGE_SPEED_INCREASE:
                    if (surrounding_objects.right_right_back_min_distance() > 12 and surrounding_objects.right_right_front_min_distance() > 3) \
                        and (surrounding_objects.right_back_min_distance() > 12 and surrounding_objects.right_front_min_distance() > 3):
                        expect_lane_idx = current_lanes.index(self.routing_target_lane) + 1
                        if expect_lane_idx in self.available_routing_index_range:
                            return surrounding_objects.right_front_object(), surrounding_objects.right_front_min_distance(), \
                                current_lanes[expect_lane_idx]
                if left_left_front_speed is not None and left_left_front_speed - front_speed > self.LANE_CHANGE_SPEED_INCREASE:
                    if (surrounding_objects.left_left_back_min_distance() > 12 and surrounding_objects.left_left_front_min_distance() > 3) \
                        and (surrounding_objects.left_back_min_distance() > 12 and surrounding_objects.left_front_min_distance() > 3):
                        expect_lane_idx = current_lanes.index(self.routing_target_lane) - 1
                        if expect_lane_idx in self.available_routing_index_range:
                            return surrounding_objects.left_front_object(), surrounding_objects.left_front_min_distance(), \
                                current_lanes[expect_lane_idx]

        # fall back to lane follow
        self.target_speed = self.NORMAL_SPEED
        self.overtake_timer += 1
        return surrounding_objects.front_object(), surrounding_objects.front_min_distance(), self.routing_target_lane

    def act_noback(self, *args, **kwargs):
        self.pdb_counter += 1
        # concat lane
        sucess = self.move_to_next_road()
        all_objects = self.control_object.lidar.get_surrounding_objects(self.control_object)

        try:
            if sucess and self.enable_lane_change:
                # perform lane change due to routing
                # self.set_target_speed(all_objects)
                acc_front_obj, acc_front_dist, steering_target_lane = self.lane_change_policy(all_objects)
            else:
                # can not find routing target lane
                surrounding_objects = RuleExpertFrontBackObjects.get_find_front_back_objs(
                    all_objects,
                    self.routing_target_lane,
                    self.control_object,
                    max_distance=self.MAX_LONG_DIST,
                    target_speed=self.NORMAL_SPEED
                )
                acc_front_obj = surrounding_objects.front_object()
                acc_front_dist = surrounding_objects.front_min_distance()
                steering_target_lane = self.routing_target_lane
        except Exception as e:
            # error fallback
            surrounding_objects = RuleExpertFrontBackObjects.get_find_front_back_objs(
                all_objects,
                self.routing_target_lane,
                self.control_object,
                max_distance=self.MAX_LONG_DIST,
                target_speed = self.NORMAL_SPEED
            )
            # surrounding_objects = RuleExpertFrontBackObjects.get_find_front_back_objs(all_objects, self.routing_target_lane, self.control_object, max_distance=self.MAX_LONG_DIST, target_speed = self.NORMAL_SPEED)
            acc_front_obj = surrounding_objects.front_object()
            acc_front_dist = surrounding_objects.front_min_distance()
            steering_target_lane = self.routing_target_lane
            logging.warning("IDM bug! fall back")
            print("IDM bug! fall back")
            print('except log {}'.format(e))

        # control by PID and IDM
        steering = self.steering_control(steering_target_lane)
        acc = self.acceleration(acc_front_obj, acc_front_dist)

        return [steering, acc]

    
    def acceleration_new(self, front_obj, dist_to_front, back_obj, dist_to_back) -> float:
        ego_vehicle = self.control_object
        ego_target_speed = not_zero(self.target_speed, 0)
        acceleration = self.ACC_FACTOR * (1 - np.power(max(ego_vehicle.speed, 0) / (ego_target_speed+0.001), self.DELTA))
        # import pdb; pdb.set_trace()
        if front_obj:
            d = dist_to_front
            dist_to_front = not_zero(dist_to_front)
            dist_to_back = not_zero(dist_to_back)
            front_speed = front_obj.speed
            ego_speed = ego_vehicle.speed
            desired_gap = self.desired_gap_new(ego_vehicle, front_obj)
            # pdb.set_trace()

            if not_zero(d) < desired_gap:
                gap_difference = desired_gap - not_zero(d)
                desired_speed_diff = gap_difference / self.TIME_WANTED_EMERGENCY
                # (desired_speed + ego_vehicle) / 2 = front_speed - desired_speed_diff
                desired_speed = (front_speed - desired_speed_diff) * 2 - ego_speed
                speed_diff = desired_speed - ego_speed
                acceleration = speed_diff * self.ACC_FACTOR

            elif not_zero(d) < desired_gap + 5:
                speed_diff = front_speed - ego_speed
                acceleration = speed_diff * self.ACC_FACTOR

            if back_obj and dist_to_back < 6:
                back_speed = back_obj.speed
                # if dist_to_front > dist_to_back and dist_to_front < desired_gap:
                # pdb.set_trace()
                if dist_to_front > dist_to_back:
                    desired_back_gap = (dist_to_back + dist_to_front) / 2
                    gap_difference = desired_back_gap - dist_to_back
                    desired_speed_diff = gap_difference / self.TIME_WANTED_EMERGENCY
                    desired_speed = min((back_speed + desired_speed_diff) * 2 - ego_speed, 35)
                    speed_diff = desired_speed - ego_speed
                    # acceleration = min(speed_diff * self.ACC_FACTOR, 3)
                    acceleration = speed_diff * self.ACC_FACTOR
                    print("acceleration: ", acceleration)
                    # pdb.set_trace()
                

        return acceleration

    def desired_gap_new(self, ego_vehicle, front_obj, projected: bool = True) -> float:
        d0 = self.DISTANCE_WANTED
        tau = self.TIME_WANTED
        ab = -self.ACC_FACTOR * self.DEACC_FACTOR
        dv = np.dot(ego_vehicle.velocity - front_obj.velocity, ego_vehicle.heading) if projected \
            else ego_vehicle.speed - front_obj.speed
        # import pdb; pdb.set_trace()
        # d_star = d0 + ego_vehicle.speed * tau + ego_vehicle.speed * dv / (2 * np.sqrt(ab))
        d_star = d0 + ego_vehicle.speed * tau
        return d_star

    def lane_change_policy(self, all_objects):
        current_lanes = self.control_object.navigation.current_ref_lanes
        if self.pdb_counter % 10 == 0:
            # import pdb; pdb.set_trace()
            surrounding_objects = RuleExpertFrontBackObjects.get_find_front_back_objs(all_objects, self.routing_target_lane, self.control_object, self.MAX_LONG_DIST, current_lanes, self.NORMAL_SPEED, debug=True)
        else:
            surrounding_objects = RuleExpertFrontBackObjects.get_find_front_back_objs(all_objects, self.routing_target_lane, self.control_object, self.MAX_LONG_DIST, current_lanes, self.NORMAL_SPEED)
        self.available_routing_index_range = [i for i in range(len(current_lanes))]
        next_lanes = self.control_object.navigation.next_ref_lanes
        lane_num_diff = len(current_lanes) - len(next_lanes) if next_lanes is not None else 0


        ''' We have to perform lane changing because the number of lanes in next road is less than current road '''
        if lane_num_diff > 0:
            print("lane num decreasing happened in left road or right road")
            # lane num decreasing happened in left road or right road
            if current_lanes[0].is_previous_lane_of(next_lanes[0]):
                index_range = [i for i in range(len(next_lanes))]
            else:
                index_range = [i for i in range(lane_num_diff, len(current_lanes))]
            self.available_routing_index_range = index_range
            if self.routing_target_lane.index[-1] not in index_range:
                # not on suitable lane do lane change !!!
                if self.routing_target_lane.index[-1] > index_range[-1]:
                    # change to left
                    if surrounding_objects.left_back_min_distance(
                    ) < self.SAFE_LANE_CHANGE_DISTANCE or surrounding_objects.left_front_min_distance() < 5:
                        # creep to wait
                        self.target_speed = self.CREEP_SPEED
                        return surrounding_objects.front_object(), surrounding_objects.front_min_distance(
                        ), self.routing_target_lane, surrounding_objects.back_object(), surrounding_objects.back_min_distance()
                    else:
                        # it is time to change lane!
                        self.target_speed = self.NORMAL_SPEED
                        return surrounding_objects.left_front_object(), surrounding_objects.left_front_min_distance(), \
                               current_lanes[self.routing_target_lane.index[-1] - 1], surrounding_objects.left_back_object(), surrounding_objects.left_back_min_distance(),
                else:
                    # change to right
                    if surrounding_objects.right_back_min_distance(
                    ) < self.SAFE_LANE_CHANGE_DISTANCE or surrounding_objects.right_front_min_distance() < 5:
                        # unsafe, creep and wait
                        self.target_speed = self.CREEP_SPEED
                        return surrounding_objects.front_object(), surrounding_objects.front_min_distance(
                        ), self.routing_target_lane, surrounding_objects.back_object(), surrounding_objects.back_min_distance()
                    else:
                        # change lane
                        self.target_speed = self.NORMAL_SPEED
                        return surrounding_objects.right_front_object(), surrounding_objects.right_front_min_distance(), \
                               current_lanes[self.routing_target_lane.index[-1] + 1], surrounding_objects.right_back_object(), surrounding_objects.right_back_min_distance(),

        ''' print surrounding object speed '''
        # print(44.0 if not surrounding_objects.left_left_lane_exist() else "{:2.1f}".format(surrounding_objects.left_left_front_min_distance()) if surrounding_objects.left_left_front_min_distance() is not None else 44.0, 44.0 if not surrounding_objects.left_lane_exist() else "{:2.1f}".format(surrounding_objects.left_front_min_distance()) if surrounding_objects.left_front_min_distance() is not None else 44.0, "{:2.1f}".format(surrounding_objects.front_min_distance()) if surrounding_objects.front_min_distance() is not None else 44.0, 44.0 if not surrounding_objects.right_lane_exist() else "{:2.1f}".format(surrounding_objects.right_front_min_distance()) if surrounding_objects.right_front_min_distance() is not None else 44.0, 44.0 if not surrounding_objects.right_right_lane_exist() else "{:2.1f}".format(surrounding_objects.right_right_front_min_distance()) if surrounding_objects.right_right_front_min_distance() is not None else 44.0)
        # print(44.0 if not surrounding_objects.left_left_lane_exist() else "{:2.1f}".format(surrounding_objects.left_left_back_min_distance()) if surrounding_objects.left_left_back_min_distance() is not None else 44.0, 44.0 if not surrounding_objects.left_lane_exist() else "{:2.1f}".format(surrounding_objects.left_back_min_distance()) if surrounding_objects.left_back_min_distance() is not None else 44.0, "{:2.1f}".format(surrounding_objects.back_min_distance()) if surrounding_objects.back_min_distance() is not None else 44.0, 44.0 if not surrounding_objects.right_lane_exist() else "{:2.1f}".format(surrounding_objects.right_back_min_distance()) if surrounding_objects.right_back_min_distance() is not None else 44.0, 44.0 if not surrounding_objects.right_right_lane_exist() else "{:2.1f}".format(surrounding_objects.right_right_back_min_distance()) if surrounding_objects.right_right_back_min_distance() is not None else 44.0)
        # print()
        # print(44.0 if not surrounding_objects.left_lane_exist() else "{:2.1f}".format(surrounding_objects.left_front_min_distance()) if surrounding_objects.left_front_min_distance() is not None else 44.0, "{:2.1f}".format(surrounding_objects.front_min_distance()) if surrounding_objects.front_min_distance() is not None else 44.0, 44.0 if not surrounding_objects.right_lane_exist() else "{:2.1f}".format(surrounding_objects.right_front_min_distance()) if surrounding_objects.right_front_min_distance() is not None else 44.0)
        # print(44.0 if not surrounding_objects.left_lane_exist() else "{:2.1f}".format(surrounding_objects.left_back_min_distance()) if surrounding_objects.left_back_min_distance() is not None else 44.0, "{:2.1f}".format(surrounding_objects.back_min_distance()) if surrounding_objects.back_min_distance() is not None else 44.0, 44.0 if not surrounding_objects.right_lane_exist() else "{:2.1f}".format(surrounding_objects.right_back_min_distance()) if surrounding_objects.right_back_min_distance() is not None else 44.0)


        ''' lane follow or active change lane/overtake for high driving speed '''
        if abs(self.control_object.speed - self.NORMAL_SPEED) > 3 and surrounding_objects.has_front_object(
        ) and abs(surrounding_objects.front_object().speed - self.NORMAL_SPEED) > 3 and self.overtake_timer > self.LANE_CHANGE_FREQ:
            # may lane change
            left_left_front_speed = surrounding_objects.left_left_front_object().speed if surrounding_objects.has_left_left_front_object() else self.MAX_SPEED \
                if surrounding_objects.left_left_lane_exist() else None
            left_front_speed = surrounding_objects.left_front_object().speed if surrounding_objects.has_left_front_object() else self.MAX_SPEED \
                if surrounding_objects.left_lane_exist() else None
            front_speed = surrounding_objects.front_object().speed if surrounding_objects.has_front_object() else self.MAX_SPEED
            right_front_speed = surrounding_objects.right_front_object().speed if surrounding_objects.has_right_front_object() else self.MAX_SPEED \
                if surrounding_objects.right_lane_exist() else None
            right_right_front_speed = surrounding_objects.right_right_front_object().speed if surrounding_objects.has_right_right_front_object() else self.MAX_SPEED \
                if surrounding_objects.right_right_lane_exist() else None
            
            # nearby lane change
            if left_front_speed is not None and (right_front_speed is None or (right_front_speed is not None and left_front_speed >= right_front_speed)):
                # consider left front first
                if left_front_speed is not None and left_front_speed - front_speed > self.LANE_CHANGE_SPEED_INCREASE:
                    if (surrounding_objects.left_back_min_distance() > 12 and surrounding_objects.left_front_min_distance() > 3):
                        expect_lane_idx = current_lanes.index(self.routing_target_lane) - 1
                        if expect_lane_idx in self.available_routing_index_range:
                            print("lange to left")
                            return surrounding_objects.left_front_object(), surrounding_objects.left_front_min_distance(), \
                                current_lanes[expect_lane_idx], surrounding_objects.left_back_object(), surrounding_objects.left_back_min_distance(),
                if right_front_speed is not None and right_front_speed - front_speed > self.LANE_CHANGE_SPEED_INCREASE:
                    if (surrounding_objects.right_back_min_distance() > 12 and surrounding_objects.right_front_min_distance() > 3):
                        expect_lane_idx = current_lanes.index(self.routing_target_lane) + 1
                        if expect_lane_idx in self.available_routing_index_range:
                            print("lange to right")
                            return surrounding_objects.right_front_object(), surrounding_objects.right_front_min_distance(), \
                                current_lanes[expect_lane_idx], surrounding_objects.right_back_object(), surrounding_objects.right_back_min_distance(),
            else:
                # consider right front first
                if right_front_speed is not None and right_front_speed - front_speed > self.LANE_CHANGE_SPEED_INCREASE:
                    if (surrounding_objects.right_back_min_distance() > 12 and surrounding_objects.right_front_min_distance() > 3):
                        expect_lane_idx = current_lanes.index(self.routing_target_lane) + 1
                        if expect_lane_idx in self.available_routing_index_range:
                            print("lange to right")
                            return surrounding_objects.right_front_object(), surrounding_objects.right_front_min_distance(), \
                                current_lanes[expect_lane_idx], surrounding_objects.right_back_object(), surrounding_objects.right_back_min_distance()
                if left_front_speed is not None and left_front_speed - front_speed > self.LANE_CHANGE_SPEED_INCREASE:
                    if (surrounding_objects.left_back_min_distance() > 12 and surrounding_objects.left_front_min_distance() > 3):
                        expect_lane_idx = current_lanes.index(self.routing_target_lane) - 1
                        if expect_lane_idx in self.available_routing_index_range:
                            print("lange to left")
                            return surrounding_objects.left_front_object(), surrounding_objects.left_front_min_distance(), \
                                current_lanes[expect_lane_idx], surrounding_objects.left_back_object(), surrounding_objects.left_back_min_distance(),

            # nearby nearby lane change
            if left_left_front_speed is not None and (right_right_front_speed is None or (right_right_front_speed is not None and left_left_front_speed >= right_right_front_speed)):
                if left_left_front_speed is not None and left_left_front_speed - front_speed > self.LANE_CHANGE_SPEED_INCREASE:
                    if (surrounding_objects.left_left_back_min_distance() > 12 and surrounding_objects.left_left_front_min_distance() > 3) \
                        and (surrounding_objects.left_back_min_distance() > 12 and surrounding_objects.left_front_min_distance() > 3):
                        expect_lane_idx = current_lanes.index(self.routing_target_lane) - 1
                        if expect_lane_idx in self.available_routing_index_range:
                            return surrounding_objects.left_front_object(), surrounding_objects.left_front_min_distance(), \
                                current_lanes[expect_lane_idx], surrounding_objects.left_back_object(), surrounding_objects.left_back_min_distance()

                if right_right_front_speed is not None and right_right_front_speed - front_speed > self.LANE_CHANGE_SPEED_INCREASE:
                    if (surrounding_objects.right_right_back_min_distance() > 12 and surrounding_objects.right_right_front_min_distance() > 3) \
                        and (surrounding_objects.right_back_min_distance() > 12 and surrounding_objects.right_front_min_distance() > 3):
                        expect_lane_idx = current_lanes.index(self.routing_target_lane) + 1
                        if expect_lane_idx in self.available_routing_index_range:
                            return surrounding_objects.right_front_object(), surrounding_objects.right_front_min_distance(), \
                                current_lanes[expect_lane_idx], surrounding_objects.right_back_object(), surrounding_objects.right_back_min_distance(),

            else:
                if right_right_front_speed is not None and right_right_front_speed - front_speed > self.LANE_CHANGE_SPEED_INCREASE:
                    if (surrounding_objects.right_right_back_min_distance() > 12 and surrounding_objects.right_right_front_min_distance() > 3) \
                        and (surrounding_objects.right_back_min_distance() > 12 and surrounding_objects.right_front_min_distance() > 3):
                        expect_lane_idx = current_lanes.index(self.routing_target_lane) + 1
                        if expect_lane_idx in self.available_routing_index_range:
                            return surrounding_objects.right_front_object(), surrounding_objects.right_front_min_distance(), \
                                current_lanes[expect_lane_idx], surrounding_objects.right_back_object(), surrounding_objects.right_back_min_distance(),
                if left_left_front_speed is not None and left_left_front_speed - front_speed > self.LANE_CHANGE_SPEED_INCREASE:
                    if (surrounding_objects.left_left_back_min_distance() > 12 and surrounding_objects.left_left_front_min_distance() > 3) \
                        and (surrounding_objects.left_back_min_distance() > 12 and surrounding_objects.left_front_min_distance() > 3):
                        expect_lane_idx = current_lanes.index(self.routing_target_lane) - 1
                        if expect_lane_idx in self.available_routing_index_range:
                            return surrounding_objects.left_front_object(), surrounding_objects.left_front_min_distance(), \
                                current_lanes[expect_lane_idx], surrounding_objects.left_back_object(), surrounding_objects.left_back_min_distance()

        # fall back to lane follow
        self.target_speed = self.NORMAL_SPEED
        self.overtake_timer += 1
        return surrounding_objects.front_object(), surrounding_objects.front_min_distance(), self.routing_target_lane, surrounding_objects.back_object(), surrounding_objects.back_min_distance()

    def act(self, *args, **kwargs):
        self.pdb_counter += 1
        # concat lane
        sucess = self.move_to_next_road()
        all_objects = self.control_object.lidar.get_surrounding_objects(self.control_object)

        try:
            if sucess and self.enable_lane_change:
                # perform lane change due to routing
                acc_front_obj, acc_front_dist, steering_target_lane, acc_back_obj, acc_back_dist = self.lane_change_policy(all_objects)
            else:
                # can not find routing target lane
                surrounding_objects = RuleExpertFrontBackObjects.get_find_front_back_objs(
                    all_objects,
                    self.routing_target_lane,
                    self.control_object,
                    max_distance=self.MAX_LONG_DIST,
                    target_speed=self.NORMAL_SPEED
                )
                acc_front_obj = surrounding_objects.front_object()
                acc_front_dist = surrounding_objects.front_min_distance()
                steering_target_lane = self.routing_target_lane
                acc_back_obj = surrounding_objects.back_object()
                acc_back_dist = surrounding_objects.back_min_distance()
        except Exception as e:
            # error fallback
            surrounding_objects = RuleExpertFrontBackObjects.get_find_front_back_objs(
                all_objects,
                self.routing_target_lane,
                self.control_object,
                max_distance=self.MAX_LONG_DIST,
                target_speed = self.NORMAL_SPEED
            )
            # surrounding_objects = RuleExpertFrontBackObjects.get_find_front_back_objs(all_objects, self.routing_target_lane, self.control_object, max_distance=self.MAX_LONG_DIST, target_speed = self.NORMAL_SPEED)
            acc_front_obj = surrounding_objects.front_object()
            acc_front_dist = surrounding_objects.front_min_distance()
            steering_target_lane = self.routing_target_lane
            acc_back_obj = surrounding_objects.back_object()
            acc_back_dist = surrounding_objects.back_min_distance()
            # acc_front_obj = None
            # acc_front_dist = 5
            # steering_target_lane = self.routing_target_lane
            logging.warning("IDM bug! fall back")
            print("IDM bug! fall back")
            print('except log {}'.format(e))

        # control by PID and IDM
        steering = self.steering_control(steering_target_lane)
        acc = self.acceleration_new(acc_front_obj, acc_front_dist, acc_back_obj, acc_back_dist)

        return [steering, acc]

class RuleExpertIDMPolicyRoundabout(IDMPolicy):
    def __init__(self, control_object, random_seed):
        super(RuleExpertIDMPolicyRoundabout, self).__init__(control_object=control_object, random_seed=random_seed)
        self.NORMAL_SPEED_CONST = 25
        self.NORMAL_SPEED = self.NORMAL_SPEED_CONST
        self.SAFE_SPEED = 15
        self.LOWEST_SPEED_FOR_RIGHTTURNNING_ON_CURVE = 2
        self.LANE_CHANGE_FREQ = 150
        self.LANE_CHANGE_SPEED_INCREASE = 2
        self.DISTANCE_WANTED = 12
        self.TIME_WANTED = 0
        self.TIME_WANTED_EMERGENCY = 0.05
        self.lateral_pid = PIDController(0.25, .002, 0.1)
        self.pdb_counter = 0
        self.entering_roundabout = False
        self.entering_roundabout_target_lane = None
        self.has_vehicle_on_conflict_lane = None

    def move_to_next_road(self):
        # routing target lane is in current ref lanes
        current_lanes = self.control_object.navigation.current_ref_lanes
        if self.routing_target_lane is None:
            self.routing_target_lane = self.control_object.lane
            return True if self.routing_target_lane in current_lanes else False

        
        # next_lanes = self.control_object.navigation.next_ref_lanes
        # if next_lanes is not None and 'circular' in str(next_lanes[0]) and 'straight' in str(current_lanes[0]):
        #     lane_length = self.control_object.lane.length
        #     current_long, _ = self.control_object.lane.local_coordinates(self.control_object.position)
        #     if lane_length - current_long < 1:
        #         self.entering_roundabout = True
        #         self.has_vehicle_on_conflict_lane = np.zeros((len(current_lanes)))
        if self.routing_target_lane not in current_lanes:
            if 'straight' in str(self.routing_target_lane) and 'circular' in str(current_lanes[0]):
                # add indicator that the vehicle just enters the roundabout
                self.entering_roundabout = True
                self.has_vehicle_on_conflict_lane = np.zeros((len(current_lanes)))
            elif self.entering_roundabout:
                # cancel enter-roundabout indicator when the vehicle is not entering the roundabout
                self.NORMAL_SPEED = 25
                self.entering_roundabout = False
                self.has_vehicle_on_conflict_lane = np.zeros((len(current_lanes)))
                
            for lane in current_lanes:
                if self.routing_target_lane.is_previous_lane_of(lane):
                    # two lanes connect
                    self.routing_target_lane = lane
                    return True
                    # lane change for lane num change
            return False
        elif self.control_object.lane in current_lanes and self.routing_target_lane is not self.control_object.lane:
            # lateral routing lane change
            self.routing_target_lane = self.control_object.lane
            self.overtake_timer = self.np_random.randint(0, int(self.LANE_CHANGE_FREQ / 2))
            return True
        else:
            return True

    def acceleration_new(self, front_obj, dist_to_front, back_obj, dist_to_back) -> float:
        ego_vehicle = self.control_object
        ego_target_speed = not_zero(self.target_speed, 0)
        acceleration = self.ACC_FACTOR * (1 - np.power(max(ego_vehicle.speed, 0) / (ego_target_speed+0.001), self.DELTA))
        # import pdb; pdb.set_trace()
        if front_obj:
            d = dist_to_front
            dist_to_front = not_zero(dist_to_front)
            dist_to_back = not_zero(dist_to_back)
            front_speed = front_obj.speed
            ego_speed = ego_vehicle.speed
            desired_gap = self.desired_gap_new(ego_vehicle, front_obj)
            # pdb.set_trace()

            if not_zero(d) < desired_gap:
                gap_difference = desired_gap - not_zero(d)
                desired_speed_diff = gap_difference / self.TIME_WANTED_EMERGENCY
                # (desired_speed + ego_vehicle) / 2 = front_speed - desired_speed_diff
                desired_speed = (front_speed - desired_speed_diff) * 2 - ego_speed
                speed_diff = desired_speed - ego_speed
                acceleration = speed_diff * self.ACC_FACTOR

            elif not_zero(d) < desired_gap + 5:
                speed_diff = front_speed - ego_speed
                acceleration = speed_diff * self.ACC_FACTOR

            if back_obj and dist_to_back < 6:
                back_speed = back_obj.speed
                # if dist_to_front > dist_to_back and dist_to_front < desired_gap:
                # pdb.set_trace()
                if dist_to_front > dist_to_back:
                    desired_back_gap = (dist_to_back + dist_to_front) / 2
                    gap_difference = desired_back_gap - dist_to_back
                    desired_speed_diff = gap_difference / self.TIME_WANTED_EMERGENCY
                    desired_speed = min((back_speed + desired_speed_diff) * 2 - ego_speed, 35)
                    speed_diff = desired_speed - ego_speed
                    # acceleration = min(speed_diff * self.ACC_FACTOR, 3)
                    acceleration = speed_diff * self.ACC_FACTOR
                    print("acceleration: ", acceleration)
                    # pdb.set_trace()
                

        return acceleration

    def desired_gap_new(self, ego_vehicle, front_obj, projected: bool = True) -> float:
        d0 = self.DISTANCE_WANTED
        tau = self.TIME_WANTED
        ab = -self.ACC_FACTOR * self.DEACC_FACTOR
        dv = np.dot(ego_vehicle.velocity - front_obj.velocity, ego_vehicle.heading) if projected \
            else ego_vehicle.speed - front_obj.speed
        # import pdb; pdb.set_trace()
        # d_star = d0 + ego_vehicle.speed * tau + ego_vehicle.speed * dv / (2 * np.sqrt(ab))
        d_star = d0 + ego_vehicle.speed * tau
        return d_star

    def lane_change_policy(self, all_objects):
        current_lanes = self.control_object.navigation.current_ref_lanes
        if self.pdb_counter % 10 == 0:
            # import pdb; pdb.set_trace()
            surrounding_objects = RuleExpertFrontBackObjects.get_find_front_back_objs(all_objects, self.routing_target_lane, self.control_object, self.MAX_LONG_DIST, current_lanes, self.NORMAL_SPEED, debug=True)
        else:
            surrounding_objects = RuleExpertFrontBackObjects.get_find_front_back_objs(all_objects, self.routing_target_lane, self.control_object, self.MAX_LONG_DIST, current_lanes, self.NORMAL_SPEED)
        self.available_routing_index_range = [i for i in range(len(current_lanes))]
        next_lanes = self.control_object.navigation.next_ref_lanes
        lane_num_diff = len(current_lanes) - len(next_lanes) if next_lanes is not None else 0

        ''' no change lane right before entering roundabout '''
        ego_vehicle = self.control_object
        if next_lanes is not None:
            if 'circular' in str(next_lanes[0]) and 'straight' in str(current_lanes[0]):
                lane_length = ego_vehicle.lane.length
                current_long, _ = ego_vehicle.lane.local_coordinates(ego_vehicle.position)
                if lane_length - current_long < 10:
                    self.target_speed = self.NORMAL_SPEED
                    return surrounding_objects.front_object(), surrounding_objects.front_min_distance(), self.routing_target_lane, surrounding_objects.back_object(), surrounding_objects.back_min_distance()

        ''' entering roundabout'''
        if next_lanes is not None:
            if self.entering_roundabout:
                ego_lane_index = current_lanes.index(self.control_object.lane)
                # rightest lane: just run
                if ego_lane_index == len(current_lanes) - 1:
                    self.target_speed = self.SAFE_SPEED
                    # self.NORMAL_SPEED = 25
                    return surrounding_objects.front_object(), surrounding_objects.front_min_distance(), self.routing_target_lane, surrounding_objects.back_object(), surrounding_objects.back_min_distance()
                # check which conflict lanes have vehicles
                self.has_vehicle_on_conflict_lane = np.zeros((len(current_lanes)))
                for one_obj in all_objects:
                    obj_lane = one_obj.lane
                    obj_ref_lanes = one_obj.navigation.current_ref_lanes
                    obj_lane_index = obj_ref_lanes.index(obj_lane)
                    for one_next_lane in self.control_object.navigation.next_ref_lanes:
                        if obj_lane.is_previous_lane_of(one_next_lane):
                            # if one_obj.position[1] < self.control_object.position[1]:
                            # if one_obj.position[1] < self.control_object.position[1]:
                            if self.control_object.lane.local_coordinates(one_obj.position)[1] < self.control_object.lane.local_coordinates(self.control_object.position)[1]:
                                # self.control_object.position[1] - one_obj.position[1]  > 
                                distance_to_ego = np.sqrt(np.sum(np.square(self.control_object.position - one_obj.position)))
                                impact_range = 20 + (len(obj_ref_lanes) - obj_lane_index - 1) * 4
                                if distance_to_ego < impact_range:
                                    self.has_vehicle_on_conflict_lane[obj_lane_index] = 1
                                
                has_vehicle_on_out_lanes = self.has_vehicle_on_conflict_lane[ego_lane_index:]
                has_vehicle_on_outer_lanes = self.has_vehicle_on_conflict_lane[ego_lane_index+1:]

                if (has_vehicle_on_out_lanes == 0).all():
                    print("all clear, just run")
                    # all clear, just run
                    # self.NORMAL_SPEED = 25
                    self.target_speed = self.NORMAL_SPEED
                    return surrounding_objects.front_object(), surrounding_objects.front_min_distance(), self.routing_target_lane, surrounding_objects.back_object(), surrounding_objects.back_min_distance()
                elif (has_vehicle_on_outer_lanes == 0).all(): 
                    print("empty on the outer lane")
                    # empty on the outer lane
                    if (surrounding_objects.right_back_min_distance() > 12 and surrounding_objects.right_front_min_distance() > 5):
                        # self.NORMAL_SPEED = 25
                        self.target_speed = self.NORMAL_SPEED
                        return surrounding_objects.right_front_object(), surrounding_objects.right_front_min_distance(), \
                                current_lanes[ego_lane_index+1], surrounding_objects.right_back_object(), surrounding_objects.right_back_min_distance(),
                    # outest_empty_lane = max(np.where(has_vehicle_on_outer_lanes == 0)[0])
                else:
                    # pdb.set_trace()
                    # if current_lanes[-1].local_coordinates(ego_vehicle.position)[1] > 0:
                        # has not enter the roundabout, speed set to 0.
                        # print("stop and wait")
                        # self.NORMAL_SPEED = 0

                        # if the car has entered the roundabout, the car would be just behind, then just run with normal speed
                    # print(current_lanes[-1].local_coordinates(ego_vehicle.position)[0])
                    # print(next_lanes[-1].local_coordinates(ego_vehicle.position)[1])
                    if next_lanes[-1].local_coordinates(ego_vehicle.position)[1] > 2:
                    # if current_lanes[-1].local_coordinates(ego_vehicle.position)[0] < 4:
                        print("stop and wait")
                        self.target_speed = 0
                        return surrounding_objects.front_object(), surrounding_objects.front_min_distance(), self.routing_target_lane, surrounding_objects.back_object(), surrounding_objects.back_min_distance()
                    

        ''' We have to perform lane changing because the number of lanes in next road is less than current road '''
        if lane_num_diff > 0:
            print("lane num decreasing happened in left road or right road")
            # lane num decreasing happened in left road or right road
            if current_lanes[0].is_previous_lane_of(next_lanes[0]):
                index_range = [i for i in range(len(next_lanes))]
            else:
                index_range = [i for i in range(lane_num_diff, len(current_lanes))]
            self.available_routing_index_range = index_range
            if self.routing_target_lane.index[-1] not in index_range:
                # not on suitable lane do lane change !!!
                if self.routing_target_lane.index[-1] > index_range[-1]:
                    # change to left
                    if surrounding_objects.left_back_min_distance(
                    ) < self.SAFE_LANE_CHANGE_DISTANCE or surrounding_objects.left_front_min_distance() < 5:
                        # creep to wait
                        self.target_speed = self.CREEP_SPEED
                        return surrounding_objects.front_object(), surrounding_objects.front_min_distance(
                        ), self.routing_target_lane, surrounding_objects.back_object(), surrounding_objects.back_min_distance()
                    else:
                        # it is time to change lane!
                        self.target_speed = self.NORMAL_SPEED
                        return surrounding_objects.left_front_object(), surrounding_objects.left_front_min_distance(), \
                               current_lanes[self.routing_target_lane.index[-1] - 1], surrounding_objects.left_back_object(), surrounding_objects.left_back_min_distance(),
                else:
                    # change to right
                    if surrounding_objects.right_back_min_distance(
                    ) < self.SAFE_LANE_CHANGE_DISTANCE or surrounding_objects.right_front_min_distance() < 5:
                        # unsafe, creep and wait
                        self.target_speed = self.CREEP_SPEED
                        return surrounding_objects.front_object(), surrounding_objects.front_min_distance(
                        ), self.routing_target_lane, surrounding_objects.back_object(), surrounding_objects.back_min_distance()
                    else:
                        # change lane
                        self.target_speed = self.NORMAL_SPEED
                        return surrounding_objects.right_front_object(), surrounding_objects.right_front_min_distance(), \
                               current_lanes[self.routing_target_lane.index[-1] + 1], surrounding_objects.right_back_object(), surrounding_objects.right_back_min_distance(),

        ''' print surrounding object speed '''
        # print(44.0 if not surrounding_objects.left_left_lane_exist() else "{:2.1f}".format(surrounding_objects.left_left_front_min_distance()) if surrounding_objects.left_left_front_min_distance() is not None else 44.0, 44.0 if not surrounding_objects.left_lane_exist() else "{:2.1f}".format(surrounding_objects.left_front_min_distance()) if surrounding_objects.left_front_min_distance() is not None else 44.0, "{:2.1f}".format(surrounding_objects.front_min_distance()) if surrounding_objects.front_min_distance() is not None else 44.0, 44.0 if not surrounding_objects.right_lane_exist() else "{:2.1f}".format(surrounding_objects.right_front_min_distance()) if surrounding_objects.right_front_min_distance() is not None else 44.0, 44.0 if not surrounding_objects.right_right_lane_exist() else "{:2.1f}".format(surrounding_objects.right_right_front_min_distance()) if surrounding_objects.right_right_front_min_distance() is not None else 44.0)
        # print(44.0 if not surrounding_objects.left_left_lane_exist() else "{:2.1f}".format(surrounding_objects.left_left_back_min_distance()) if surrounding_objects.left_left_back_min_distance() is not None else 44.0, 44.0 if not surrounding_objects.left_lane_exist() else "{:2.1f}".format(surrounding_objects.left_back_min_distance()) if surrounding_objects.left_back_min_distance() is not None else 44.0, "{:2.1f}".format(surrounding_objects.back_min_distance()) if surrounding_objects.back_min_distance() is not None else 44.0, 44.0 if not surrounding_objects.right_lane_exist() else "{:2.1f}".format(surrounding_objects.right_back_min_distance()) if surrounding_objects.right_back_min_distance() is not None else 44.0, 44.0 if not surrounding_objects.right_right_lane_exist() else "{:2.1f}".format(surrounding_objects.right_right_back_min_distance()) if surrounding_objects.right_right_back_min_distance() is not None else 44.0)
        # print("speed: ", self.control_object.last_spd)
        # print()
        # print(44.0 if not surrounding_objects.left_lane_exist() else "{:2.1f}".format(surrounding_objects.left_front_min_distance()) if surrounding_objects.left_front_min_distance() is not None else 44.0, "{:2.1f}".format(surrounding_objects.front_min_distance()) if surrounding_objects.front_min_distance() is not None else 44.0, 44.0 if not surrounding_objects.right_lane_exist() else "{:2.1f}".format(surrounding_objects.right_front_min_distance()) if surrounding_objects.right_front_min_distance() is not None else 44.0)
        # print(44.0 if not surrounding_objects.left_lane_exist() else "{:2.1f}".format(surrounding_objects.left_back_min_distance()) if surrounding_objects.left_back_min_distance() is not None else 44.0, "{:2.1f}".format(surrounding_objects.back_min_distance()) if surrounding_objects.back_min_distance() is not None else 44.0, 44.0 if not surrounding_objects.right_lane_exist() else "{:2.1f}".format(surrounding_objects.right_back_min_distance()) if surrounding_objects.right_back_min_distance() is not None else 44.0)


        ''' lane follow or active change lane/overtake for high driving speed '''
        if abs(self.control_object.speed - self.NORMAL_SPEED) > 3 and surrounding_objects.has_front_object(
         ) and abs(surrounding_objects.front_object().speed - self.NORMAL_SPEED) > 3 and self.overtake_timer > self.LANE_CHANGE_FREQ:
            # may lane change
            left_left_front_speed = surrounding_objects.left_left_front_object().speed if surrounding_objects.has_left_left_front_object() else self.MAX_SPEED \
                if surrounding_objects.left_left_lane_exist() else None
            left_front_speed = surrounding_objects.left_front_object().speed if surrounding_objects.has_left_front_object() else self.MAX_SPEED \
                if surrounding_objects.left_lane_exist() else None
            front_speed = surrounding_objects.front_object().speed if surrounding_objects.has_front_object() else self.MAX_SPEED
            right_front_speed = surrounding_objects.right_front_object().speed if surrounding_objects.has_right_front_object() else self.MAX_SPEED \
                if surrounding_objects.right_lane_exist() else None
            right_right_front_speed = surrounding_objects.right_right_front_object().speed if surrounding_objects.has_right_right_front_object() else self.MAX_SPEED \
                if surrounding_objects.right_right_lane_exist() else None
            
            # nearby lane change
            if left_front_speed is not None and (right_front_speed is None or (right_front_speed is not None and left_front_speed >= right_front_speed)):
                # consider left front first
                if left_front_speed is not None and left_front_speed - front_speed > self.LANE_CHANGE_SPEED_INCREASE:
                    if (surrounding_objects.left_back_min_distance() > 12 and surrounding_objects.left_front_min_distance() > 5):
                        expect_lane_idx = current_lanes.index(self.routing_target_lane) - 1
                        if expect_lane_idx in self.available_routing_index_range:
                            print("lange to left")
                            self.target_speed = self.SAFE_SPEED
                            return surrounding_objects.left_front_object(), surrounding_objects.left_front_min_distance(), \
                                current_lanes[expect_lane_idx], surrounding_objects.left_back_object(), surrounding_objects.left_back_min_distance(),
                if right_front_speed is not None and right_front_speed - front_speed > self.LANE_CHANGE_SPEED_INCREASE:
                    if (surrounding_objects.right_back_min_distance() > 12 and surrounding_objects.right_front_min_distance() > 5):
                        expect_lane_idx = current_lanes.index(self.routing_target_lane) + 1
                        if expect_lane_idx in self.available_routing_index_range:
                            if 'circular' in str(current_lanes[0]): # avoid over turning to hit road boundary
                                if self.control_object.last_spd > self.LOWEST_SPEED_FOR_RIGHTTURNNING_ON_CURVE:  # no turning at low speed
                                    print("lange to right")
                                    self.target_speed = self.SAFE_SPEED  # slow down for the turning
                                    return surrounding_objects.right_front_object(), surrounding_objects.right_front_min_distance(), \
                                        current_lanes[expect_lane_idx], surrounding_objects.right_back_object(), surrounding_objects.right_back_min_distance(),
                            else: 
                                print("lange to right")
                                self.target_speed = self.SAFE_SPEED
                                return surrounding_objects.right_front_object(), surrounding_objects.right_front_min_distance(), \
                                    current_lanes[expect_lane_idx], surrounding_objects.right_back_object(), surrounding_objects.right_back_min_distance(),
            else:
                # consider right front first
                if right_front_speed is not None and right_front_speed - front_speed > self.LANE_CHANGE_SPEED_INCREASE:
                    if (surrounding_objects.right_back_min_distance() > 12 and surrounding_objects.right_front_min_distance() > 5):
                        expect_lane_idx = current_lanes.index(self.routing_target_lane) + 1
                        if expect_lane_idx in self.available_routing_index_range:
                            if 'circular' in str(current_lanes[0]): # avoid over turning to hit road boundary
                                if self.control_object.last_spd > self.LOWEST_SPEED_FOR_RIGHTTURNNING_ON_CURVE:  # no turning at low speed
                                    print("lange to right")
                                    self.target_speed = self.SAFE_SPEED  # slow down for the turning
                                    return surrounding_objects.right_front_object(), surrounding_objects.right_front_min_distance(), \
                                        current_lanes[expect_lane_idx], surrounding_objects.right_back_object(), surrounding_objects.right_back_min_distance(),
                            else: 
                                print("lange to right")
                                self.target_speed = self.SAFE_SPEED
                                return surrounding_objects.right_front_object(), surrounding_objects.right_front_min_distance(), \
                                    current_lanes[expect_lane_idx], surrounding_objects.right_back_object(), surrounding_objects.right_back_min_distance(),
                if left_front_speed is not None and left_front_speed - front_speed > self.LANE_CHANGE_SPEED_INCREASE:
                    if (surrounding_objects.left_back_min_distance() > 12 and surrounding_objects.left_front_min_distance() > 5):
                        expect_lane_idx = current_lanes.index(self.routing_target_lane) - 1
                        if expect_lane_idx in self.available_routing_index_range:
                            print("lange to left")
                            self.target_speed = self.SAFE_SPEED
                            return surrounding_objects.left_front_object(), surrounding_objects.left_front_min_distance(), \
                                current_lanes[expect_lane_idx], surrounding_objects.left_back_object(), surrounding_objects.left_back_min_distance(),

            # nearby nearby lane change
            if left_left_front_speed is not None and (right_right_front_speed is None or (right_right_front_speed is not None and left_left_front_speed >= right_right_front_speed)):
                if left_left_front_speed is not None and left_left_front_speed - front_speed > self.LANE_CHANGE_SPEED_INCREASE:
                    if (surrounding_objects.left_left_back_min_distance() > 12 and surrounding_objects.left_left_front_min_distance() > 5) \
                        and (surrounding_objects.left_back_min_distance() > 12 and surrounding_objects.left_front_min_distance() > 5):
                        expect_lane_idx = current_lanes.index(self.routing_target_lane) - 1
                        if expect_lane_idx in self.available_routing_index_range:
                            print("lange to left")
                            self.target_speed = self.SAFE_SPEED
                            return surrounding_objects.left_front_object(), surrounding_objects.left_front_min_distance(), \
                                current_lanes[expect_lane_idx], surrounding_objects.left_back_object(), surrounding_objects.left_back_min_distance()

                if right_right_front_speed is not None and right_right_front_speed - front_speed > self.LANE_CHANGE_SPEED_INCREASE:
                    if (surrounding_objects.right_right_back_min_distance() > 12 and surrounding_objects.right_right_front_min_distance() > 3) \
                        and (surrounding_objects.right_back_min_distance() > 12 and surrounding_objects.right_front_min_distance() > 5):
                        expect_lane_idx = current_lanes.index(self.routing_target_lane) + 1
                        if expect_lane_idx in self.available_routing_index_range:
                            if 'circular' in str(current_lanes[0]): # avoid over turning to hit road boundary
                                if self.control_object.last_spd > self.LOWEST_SPEED_FOR_RIGHTTURNNING_ON_CURVE:  # no turning at low speed
                                    print("lange to right")
                                    self.target_speed = self.SAFE_SPEED  # slow down for the turning
                                    return surrounding_objects.right_front_object(), surrounding_objects.right_front_min_distance(), \
                                        current_lanes[expect_lane_idx], surrounding_objects.right_back_object(), surrounding_objects.right_back_min_distance(),
                            else: 
                                print("lange to right")
                                self.target_speed = self.SAFE_SPEED
                                return surrounding_objects.right_front_object(), surrounding_objects.right_front_min_distance(), \
                                    current_lanes[expect_lane_idx], surrounding_objects.right_back_object(), surrounding_objects.right_back_min_distance(),

            else:
                if right_right_front_speed is not None and right_right_front_speed - front_speed > self.LANE_CHANGE_SPEED_INCREASE:
                    if (surrounding_objects.right_right_back_min_distance() > 12 and surrounding_objects.right_right_front_min_distance() > 3) \
                        and (surrounding_objects.right_back_min_distance() > 12 and surrounding_objects.right_front_min_distance() > 5):
                        expect_lane_idx = current_lanes.index(self.routing_target_lane) + 1
                        if expect_lane_idx in self.available_routing_index_range:
                            if 'circular' in str(current_lanes[0]): # avoid over turning to hit road boundary
                                if self.control_object.last_spd > self.LOWEST_SPEED_FOR_RIGHTTURNNING_ON_CURVE:  # no turning at low speed
                                    print("lange to right")
                                    self.target_speed = self.SAFE_SPEED  # slow down for the turning
                                    return surrounding_objects.right_front_object(), surrounding_objects.right_front_min_distance(), \
                                        current_lanes[expect_lane_idx], surrounding_objects.right_back_object(), surrounding_objects.right_back_min_distance(),
                            else: 
                                print("lange to right")
                                self.target_speed = self.SAFE_SPEED
                                return surrounding_objects.right_front_object(), surrounding_objects.right_front_min_distance(), \
                                    current_lanes[expect_lane_idx], surrounding_objects.right_back_object(), surrounding_objects.right_back_min_distance(),
                if left_left_front_speed is not None and left_left_front_speed - front_speed > self.LANE_CHANGE_SPEED_INCREASE:
                    if (surrounding_objects.left_left_back_min_distance() > 12 and surrounding_objects.left_left_front_min_distance() > 5) \
                        and (surrounding_objects.left_back_min_distance() > 12 and surrounding_objects.left_front_min_distance() > 5):
                        expect_lane_idx = current_lanes.index(self.routing_target_lane) - 1
                        if expect_lane_idx in self.available_routing_index_range:
                            print("lange to left")
                            self.target_speed = self.SAFE_SPEED
                            return surrounding_objects.left_front_object(), surrounding_objects.left_front_min_distance(), \
                                current_lanes[expect_lane_idx], surrounding_objects.left_back_object(), surrounding_objects.left_back_min_distance()

        ''' leaving roundabout, slow down '''
        if 'circular' in str(current_lanes[0]) and 'straight' in str(next_lanes[0]):
            self.target_speed = self.SAFE_SPEED
            self.overtake_timer += 1
            return surrounding_objects.front_object(), surrounding_objects.front_min_distance(), self.routing_target_lane, surrounding_objects.back_object(), surrounding_objects.back_min_distance()



        # fall back to lane follow
        self.target_speed = self.NORMAL_SPEED
        self.overtake_timer += 1
        return surrounding_objects.front_object(), surrounding_objects.front_min_distance(), self.routing_target_lane, surrounding_objects.back_object(), surrounding_objects.back_min_distance()

    def act(self, *args, **kwargs):
        self.pdb_counter += 1
        # concat lane
        sucess = self.move_to_next_road()
        all_objects = self.control_object.lidar.get_surrounding_objects(self.control_object)

        try:
            if sucess and self.enable_lane_change:
                # perform lane change due to routing
                acc_front_obj, acc_front_dist, steering_target_lane, acc_back_obj, acc_back_dist = self.lane_change_policy(all_objects)
            else:
                # can not find routing target lane
                surrounding_objects = RuleExpertFrontBackObjects.get_find_front_back_objs(
                    all_objects,
                    self.routing_target_lane,
                    self.control_object,
                    max_distance=self.MAX_LONG_DIST,
                    target_speed=self.NORMAL_SPEED
                )
                acc_front_obj = surrounding_objects.front_object()
                acc_front_dist = surrounding_objects.front_min_distance()
                steering_target_lane = self.routing_target_lane
                acc_back_obj = surrounding_objects.back_object()
                acc_back_dist = surrounding_objects.back_min_distance()
        except Exception as e:
            # error fallback
            surrounding_objects = RuleExpertFrontBackObjects.get_find_front_back_objs(
                all_objects,
                self.routing_target_lane,
                self.control_object,
                max_distance=self.MAX_LONG_DIST,
                target_speed = self.NORMAL_SPEED
            )
            # surrounding_objects = RuleExpertFrontBackObjects.get_find_front_back_objs(all_objects, self.routing_target_lane, self.control_object, max_distance=self.MAX_LONG_DIST, target_speed = self.NORMAL_SPEED)
            acc_front_obj = surrounding_objects.front_object()
            acc_front_dist = surrounding_objects.front_min_distance()
            steering_target_lane = self.routing_target_lane
            acc_back_obj = surrounding_objects.back_object()
            acc_back_dist = surrounding_objects.back_min_distance()
            # acc_front_obj = None
            # acc_front_dist = 5
            # steering_target_lane = self.routing_target_lane
            logging.warning("IDM bug! fall back")
            print("IDM bug! fall back")
            print('except log {}'.format(e))

        # control by PID and IDM
        steering = self.steering_control(steering_target_lane)
        acc = self.acceleration_new(acc_front_obj, acc_front_dist, acc_back_obj, acc_back_dist)

        # ego_vehicle = self.control_object
        # last_position = ego_vehicle.last_macro_position
        # current_position = ego_vehicle.position

        # import numpy as np
        # ego_vehicle.last_spd = np.sqrt(np.sum(np.square(current_position - last_position))) / ego_vehicle.physics_world_step_size
        # print("last_position: ", current_position)
        # print("current_position: ", current_position)
        # print("ego_vehicle.last_spd: ", ego_vehicle.last_spd)
        # import pdb; pdb.set_trace()
        # ego_vehicle.last_spd = norm / ego_vehicle.physics_world_step_size

        return [steering, acc]

class RuleExpertIDMPolicyRoundabout_simple(IDMPolicy):
    def __init__(self, control_object, random_seed):
        super(RuleExpertIDMPolicyRoundabout_simple, self).__init__(control_object=control_object, random_seed=random_seed)
        self.NORMAL_SPEED_CONST = 25
        self.NORMAL_SPEED = self.NORMAL_SPEED_CONST
        self.LANE_CHANGE_FREQ = 150
        self.LANE_CHANGE_SPEED_INCREASE = 2
        self.DISTANCE_WANTED = 12
        self.TIME_WANTED = 0
        self.TIME_WANTED_EMERGENCY = 0.05
        self.lateral_pid = PIDController(0.25, .002, 0.1)
        self.pdb_counter = 0
        self.entering_roundabout = False
        self.entering_roundabout_target_lane = None
        self.has_vehicle_on_conflict_lane = None

    def move_to_next_road(self):
        # routing target lane is in current ref lanes
        current_lanes = self.control_object.navigation.current_ref_lanes
        if self.routing_target_lane is None:
            self.routing_target_lane = self.control_object.lane
            return True if self.routing_target_lane in current_lanes else False
        if self.routing_target_lane not in current_lanes:
            if 'straight' in str(self.routing_target_lane) and 'circular' in str(current_lanes[0]):
                # add indicator that the vehicle just enters the roundabout
                self.entering_roundabout = True
                self.has_vehicle_on_conflict_lane = np.zeros((len(current_lanes)))
            elif self.entering_roundabout:
                # cancel enter-roundabout indicator when the vehicle is not entering the roundabout
                self.NORMAL_SPEED = 25
                self.entering_roundabout = False
                self.has_vehicle_on_conflict_lane = np.zeros((len(current_lanes)))
                
            for lane in current_lanes:
                if self.routing_target_lane.is_previous_lane_of(lane):
                    # two lanes connect
                    self.routing_target_lane = lane
                    return True
                    # lane change for lane num change
            return False
        elif self.control_object.lane in current_lanes and self.routing_target_lane is not self.control_object.lane:
            # lateral routing lane change
            self.routing_target_lane = self.control_object.lane
            self.overtake_timer = self.np_random.randint(0, int(self.LANE_CHANGE_FREQ / 2))
            return True
        else:
            return True

    def acceleration_new(self, front_obj, dist_to_front, back_obj, dist_to_back) -> float:
        ego_vehicle = self.control_object
        ego_target_speed = not_zero(self.target_speed, 0)
        acceleration = self.ACC_FACTOR * (1 - np.power(max(ego_vehicle.speed, 0) / (ego_target_speed+0.001), self.DELTA))
        # import pdb; pdb.set_trace()
        if front_obj:
            d = dist_to_front
            dist_to_front = not_zero(dist_to_front)
            dist_to_back = not_zero(dist_to_back)
            front_speed = front_obj.speed
            ego_speed = ego_vehicle.speed
            desired_gap = self.desired_gap_new(ego_vehicle, front_obj)
            # pdb.set_trace()

            if not_zero(d) < desired_gap:
                gap_difference = desired_gap - not_zero(d)
                desired_speed_diff = gap_difference / self.TIME_WANTED_EMERGENCY
                # (desired_speed + ego_vehicle) / 2 = front_speed - desired_speed_diff
                desired_speed = (front_speed - desired_speed_diff) * 2 - ego_speed
                speed_diff = desired_speed - ego_speed
                acceleration = speed_diff * self.ACC_FACTOR

            elif not_zero(d) < desired_gap + 5:
                speed_diff = front_speed - ego_speed
                acceleration = speed_diff * self.ACC_FACTOR

            if back_obj and dist_to_back < 6:
                back_speed = back_obj.speed
                # if dist_to_front > dist_to_back and dist_to_front < desired_gap:
                # pdb.set_trace()
                if dist_to_front > dist_to_back:
                    desired_back_gap = (dist_to_back + dist_to_front) / 2
                    gap_difference = desired_back_gap - dist_to_back
                    desired_speed_diff = gap_difference / self.TIME_WANTED_EMERGENCY
                    desired_speed = min((back_speed + desired_speed_diff) * 2 - ego_speed, 35)
                    speed_diff = desired_speed - ego_speed
                    # acceleration = min(speed_diff * self.ACC_FACTOR, 3)
                    acceleration = speed_diff * self.ACC_FACTOR
                    print("acceleration: ", acceleration)
                    # pdb.set_trace()
                

        return acceleration

    def desired_gap_new(self, ego_vehicle, front_obj, projected: bool = True) -> float:
        d0 = self.DISTANCE_WANTED
        tau = self.TIME_WANTED
        ab = -self.ACC_FACTOR * self.DEACC_FACTOR
        dv = np.dot(ego_vehicle.velocity - front_obj.velocity, ego_vehicle.heading) if projected \
            else ego_vehicle.speed - front_obj.speed
        # import pdb; pdb.set_trace()
        # d_star = d0 + ego_vehicle.speed * tau + ego_vehicle.speed * dv / (2 * np.sqrt(ab))
        d_star = d0 + ego_vehicle.speed * tau
        return d_star

    def lane_change_policy(self, all_objects):
        current_lanes = self.control_object.navigation.current_ref_lanes
        if self.pdb_counter % 10 == 0:
            # import pdb; pdb.set_trace()
            surrounding_objects = RuleExpertFrontBackObjects.get_find_front_back_objs(all_objects, self.routing_target_lane, self.control_object, self.MAX_LONG_DIST, current_lanes, self.NORMAL_SPEED, debug=True)
        else:
            surrounding_objects = RuleExpertFrontBackObjects.get_find_front_back_objs(all_objects, self.routing_target_lane, self.control_object, self.MAX_LONG_DIST, current_lanes, self.NORMAL_SPEED)
        self.available_routing_index_range = [i for i in range(len(current_lanes))]
        next_lanes = self.control_object.navigation.next_ref_lanes
        lane_num_diff = len(current_lanes) - len(next_lanes) if next_lanes is not None else 0

        ''' no change lane right before entering roundabout '''
        ego_vehicle = self.control_object
        if next_lanes is not None:
            if 'circular' in str(next_lanes[0]) and 'straight' in str(current_lanes[0]):
                lane_length = ego_vehicle.lane.length
                current_long, _ = ego_vehicle.lane.local_coordinates(ego_vehicle.position)
                if lane_length - current_long < 10:
                    return surrounding_objects.front_object(), surrounding_objects.front_min_distance(), self.routing_target_lane, surrounding_objects.back_object(), surrounding_objects.back_min_distance()

        ''' entering roundabout'''
        if next_lanes is not None:
            if self.entering_roundabout:
                ego_lane_index = current_lanes.index(self.control_object.lane)
                # rightest lane: just run
                if ego_lane_index == len(current_lanes) - 1:
                    self.NORMAL_SPEED = 25
                    return surrounding_objects.front_object(), surrounding_objects.front_min_distance(), self.routing_target_lane, surrounding_objects.back_object(), surrounding_objects.back_min_distance()
                # check which conflict lanes have vehicles
                self.has_vehicle_on_conflict_lane = np.zeros((len(current_lanes)))
                for one_obj in all_objects:
                    obj_lane = one_obj.lane
                    obj_ref_lanes = one_obj.navigation.current_ref_lanes
                    obj_lane_index = obj_ref_lanes.index(obj_lane)
                    for one_next_lane in self.control_object.navigation.next_ref_lanes:
                        if obj_lane.is_previous_lane_of(one_next_lane):
                            if one_obj.position[1] < self.control_object.position[1]:
                                # self.control_object.position[1] - one_obj.position[1]  > 
                                distance_to_ego = np.sqrt(np.sum(np.square(self.control_object.position - one_obj.position)))
                                impact_range = 20 + (len(obj_ref_lanes) - obj_lane_index - 1) * 4
                                if distance_to_ego < impact_range:
                                    self.has_vehicle_on_conflict_lane[obj_lane_index] = 1
                                
                has_vehicle_on_out_lanes = self.has_vehicle_on_conflict_lane[ego_lane_index:]
                has_vehicle_on_outer_lanes = self.has_vehicle_on_conflict_lane[ego_lane_index+1:]

                if (has_vehicle_on_out_lanes == 0).all():
                    print("all clear, just run")
                    # all clear, just run
                    self.NORMAL_SPEED = 25
                    return surrounding_objects.front_object(), surrounding_objects.front_min_distance(), self.routing_target_lane, surrounding_objects.back_object(), surrounding_objects.back_min_distance()
                elif (has_vehicle_on_outer_lanes == 0).all(): 
                    print("empty on the outer lane")
                    # empty on the outer lane
                    if (surrounding_objects.right_back_min_distance() > 12 and surrounding_objects.right_front_min_distance() > 3):
                        self.NORMAL_SPEED = 25
                        return surrounding_objects.right_front_object(), surrounding_objects.right_front_min_distance(), \
                                current_lanes[ego_lane_index+1], surrounding_objects.right_back_object(), surrounding_objects.right_back_min_distance(),
                    # outest_empty_lane = max(np.where(has_vehicle_on_outer_lanes == 0)[0])
                else:
                    # pdb.set_trace()
                    if current_lanes[-1].local_coordinates(ego_vehicle.position)[1] > 0:
                        # has not enter the roundabout, speed set to 0.
                        print("stop and wait")
                        self.NORMAL_SPEED = 0
                        # if the car has entered the roundabout, the car would be just behind, then just run with normal speed
                    return surrounding_objects.front_object(), surrounding_objects.front_min_distance(), self.routing_target_lane, surrounding_objects.back_object(), surrounding_objects.back_min_distance()

        ''' We have to perform lane changing because the number of lanes in next road is less than current road '''
        if lane_num_diff > 0:
            print("lane num decreasing happened in left road or right road")
            # lane num decreasing happened in left road or right road
            if current_lanes[0].is_previous_lane_of(next_lanes[0]):
                index_range = [i for i in range(len(next_lanes))]
            else:
                index_range = [i for i in range(lane_num_diff, len(current_lanes))]
            self.available_routing_index_range = index_range
            if self.routing_target_lane.index[-1] not in index_range:
                # not on suitable lane do lane change !!!
                if self.routing_target_lane.index[-1] > index_range[-1]:
                    # change to left
                    if surrounding_objects.left_back_min_distance(
                    ) < self.SAFE_LANE_CHANGE_DISTANCE or surrounding_objects.left_front_min_distance() < 5:
                        # creep to wait
                        self.target_speed = self.CREEP_SPEED
                        return surrounding_objects.front_object(), surrounding_objects.front_min_distance(
                        ), self.routing_target_lane, surrounding_objects.back_object(), surrounding_objects.back_min_distance()
                    else:
                        # it is time to change lane!
                        self.target_speed = self.NORMAL_SPEED
                        return surrounding_objects.left_front_object(), surrounding_objects.left_front_min_distance(), \
                               current_lanes[self.routing_target_lane.index[-1] - 1], surrounding_objects.left_back_object(), surrounding_objects.left_back_min_distance(),
                else:
                    # change to right
                    if surrounding_objects.right_back_min_distance(
                    ) < self.SAFE_LANE_CHANGE_DISTANCE or surrounding_objects.right_front_min_distance() < 5:
                        # unsafe, creep and wait
                        self.target_speed = self.CREEP_SPEED
                        return surrounding_objects.front_object(), surrounding_objects.front_min_distance(
                        ), self.routing_target_lane, surrounding_objects.back_object(), surrounding_objects.back_min_distance()
                    else:
                        # change lane
                        self.target_speed = self.NORMAL_SPEED
                        return surrounding_objects.right_front_object(), surrounding_objects.right_front_min_distance(), \
                               current_lanes[self.routing_target_lane.index[-1] + 1], surrounding_objects.right_back_object(), surrounding_objects.right_back_min_distance(),

        ''' print surrounding object speed '''
        # print(44.0 if not surrounding_objects.left_left_lane_exist() else "{:2.1f}".format(surrounding_objects.left_left_front_min_distance()) if surrounding_objects.left_left_front_min_distance() is not None else 44.0, 44.0 if not surrounding_objects.left_lane_exist() else "{:2.1f}".format(surrounding_objects.left_front_min_distance()) if surrounding_objects.left_front_min_distance() is not None else 44.0, "{:2.1f}".format(surrounding_objects.front_min_distance()) if surrounding_objects.front_min_distance() is not None else 44.0, 44.0 if not surrounding_objects.right_lane_exist() else "{:2.1f}".format(surrounding_objects.right_front_min_distance()) if surrounding_objects.right_front_min_distance() is not None else 44.0, 44.0 if not surrounding_objects.right_right_lane_exist() else "{:2.1f}".format(surrounding_objects.right_right_front_min_distance()) if surrounding_objects.right_right_front_min_distance() is not None else 44.0)
        # print(44.0 if not surrounding_objects.left_left_lane_exist() else "{:2.1f}".format(surrounding_objects.left_left_back_min_distance()) if surrounding_objects.left_left_back_min_distance() is not None else 44.0, 44.0 if not surrounding_objects.left_lane_exist() else "{:2.1f}".format(surrounding_objects.left_back_min_distance()) if surrounding_objects.left_back_min_distance() is not None else 44.0, "{:2.1f}".format(surrounding_objects.back_min_distance()) if surrounding_objects.back_min_distance() is not None else 44.0, 44.0 if not surrounding_objects.right_lane_exist() else "{:2.1f}".format(surrounding_objects.right_back_min_distance()) if surrounding_objects.right_back_min_distance() is not None else 44.0, 44.0 if not surrounding_objects.right_right_lane_exist() else "{:2.1f}".format(surrounding_objects.right_right_back_min_distance()) if surrounding_objects.right_right_back_min_distance() is not None else 44.0)
        # print()
        # print(44.0 if not surrounding_objects.left_lane_exist() else "{:2.1f}".format(surrounding_objects.left_front_min_distance()) if surrounding_objects.left_front_min_distance() is not None else 44.0, "{:2.1f}".format(surrounding_objects.front_min_distance()) if surrounding_objects.front_min_distance() is not None else 44.0, 44.0 if not surrounding_objects.right_lane_exist() else "{:2.1f}".format(surrounding_objects.right_front_min_distance()) if surrounding_objects.right_front_min_distance() is not None else 44.0)
        # print(44.0 if not surrounding_objects.left_lane_exist() else "{:2.1f}".format(surrounding_objects.left_back_min_distance()) if surrounding_objects.left_back_min_distance() is not None else 44.0, "{:2.1f}".format(surrounding_objects.back_min_distance()) if surrounding_objects.back_min_distance() is not None else 44.0, 44.0 if not surrounding_objects.right_lane_exist() else "{:2.1f}".format(surrounding_objects.right_back_min_distance()) if surrounding_objects.right_back_min_distance() is not None else 44.0)


        ''' lane follow or active change lane/overtake for high driving speed '''
        if abs(self.control_object.speed - self.NORMAL_SPEED) > 3 and surrounding_objects.has_front_object(
        ) and abs(surrounding_objects.front_object().speed - self.NORMAL_SPEED) > 3 and self.overtake_timer > self.LANE_CHANGE_FREQ:
            # may lane change
            left_left_front_speed = surrounding_objects.left_left_front_object().speed if surrounding_objects.has_left_left_front_object() else self.MAX_SPEED \
                if surrounding_objects.left_left_lane_exist() else None
            left_front_speed = surrounding_objects.left_front_object().speed if surrounding_objects.has_left_front_object() else self.MAX_SPEED \
                if surrounding_objects.left_lane_exist() else None
            front_speed = surrounding_objects.front_object().speed if surrounding_objects.has_front_object() else self.MAX_SPEED
            right_front_speed = surrounding_objects.right_front_object().speed if surrounding_objects.has_right_front_object() else self.MAX_SPEED \
                if surrounding_objects.right_lane_exist() else None
            right_right_front_speed = surrounding_objects.right_right_front_object().speed if surrounding_objects.has_right_right_front_object() else self.MAX_SPEED \
                if surrounding_objects.right_right_lane_exist() else None
            
            # nearby lane change
            if left_front_speed is not None and (right_front_speed is None or (right_front_speed is not None and left_front_speed >= right_front_speed)):
                # consider left front first
                if left_front_speed is not None and left_front_speed - front_speed > self.LANE_CHANGE_SPEED_INCREASE:
                    if (surrounding_objects.left_back_min_distance() > 12 and surrounding_objects.left_front_min_distance() > 3):
                        expect_lane_idx = current_lanes.index(self.routing_target_lane) - 1
                        if expect_lane_idx in self.available_routing_index_range:
                            print("lange to left")
                            return surrounding_objects.left_front_object(), surrounding_objects.left_front_min_distance(), \
                                current_lanes[expect_lane_idx], surrounding_objects.left_back_object(), surrounding_objects.left_back_min_distance(),
                if right_front_speed is not None and right_front_speed - front_speed > self.LANE_CHANGE_SPEED_INCREASE:
                    if (surrounding_objects.right_back_min_distance() > 12 and surrounding_objects.right_front_min_distance() > 3):
                        expect_lane_idx = current_lanes.index(self.routing_target_lane) + 1
                        if expect_lane_idx in self.available_routing_index_range:
                            print("lange to right")
                            return surrounding_objects.right_front_object(), surrounding_objects.right_front_min_distance(), \
                                current_lanes[expect_lane_idx], surrounding_objects.right_back_object(), surrounding_objects.right_back_min_distance(),
            else:
                # consider right front first
                if right_front_speed is not None and right_front_speed - front_speed > self.LANE_CHANGE_SPEED_INCREASE:
                    if (surrounding_objects.right_back_min_distance() > 12 and surrounding_objects.right_front_min_distance() > 3):
                        expect_lane_idx = current_lanes.index(self.routing_target_lane) + 1
                        if expect_lane_idx in self.available_routing_index_range:
                            print("lange to right")
                            return surrounding_objects.right_front_object(), surrounding_objects.right_front_min_distance(), \
                                current_lanes[expect_lane_idx], surrounding_objects.right_back_object(), surrounding_objects.right_back_min_distance()
                if left_front_speed is not None and left_front_speed - front_speed > self.LANE_CHANGE_SPEED_INCREASE:
                    if (surrounding_objects.left_back_min_distance() > 12 and surrounding_objects.left_front_min_distance() > 3):
                        expect_lane_idx = current_lanes.index(self.routing_target_lane) - 1
                        if expect_lane_idx in self.available_routing_index_range:
                            print("lange to left")
                            return surrounding_objects.left_front_object(), surrounding_objects.left_front_min_distance(), \
                                current_lanes[expect_lane_idx], surrounding_objects.left_back_object(), surrounding_objects.left_back_min_distance(),

            # nearby nearby lane change
            if left_left_front_speed is not None and (right_right_front_speed is None or (right_right_front_speed is not None and left_left_front_speed >= right_right_front_speed)):
                if left_left_front_speed is not None and left_left_front_speed - front_speed > self.LANE_CHANGE_SPEED_INCREASE:
                    if (surrounding_objects.left_left_back_min_distance() > 12 and surrounding_objects.left_left_front_min_distance() > 3) \
                        and (surrounding_objects.left_back_min_distance() > 12 and surrounding_objects.left_front_min_distance() > 3):
                        expect_lane_idx = current_lanes.index(self.routing_target_lane) - 1
                        if expect_lane_idx in self.available_routing_index_range:
                            return surrounding_objects.left_front_object(), surrounding_objects.left_front_min_distance(), \
                                current_lanes[expect_lane_idx], surrounding_objects.left_back_object(), surrounding_objects.left_back_min_distance()

                if right_right_front_speed is not None and right_right_front_speed - front_speed > self.LANE_CHANGE_SPEED_INCREASE:
                    if (surrounding_objects.right_right_back_min_distance() > 12 and surrounding_objects.right_right_front_min_distance() > 3) \
                        and (surrounding_objects.right_back_min_distance() > 12 and surrounding_objects.right_front_min_distance() > 3):
                        expect_lane_idx = current_lanes.index(self.routing_target_lane) + 1
                        if expect_lane_idx in self.available_routing_index_range:
                            return surrounding_objects.right_front_object(), surrounding_objects.right_front_min_distance(), \
                                current_lanes[expect_lane_idx], surrounding_objects.right_back_object(), surrounding_objects.right_back_min_distance(),

            else:
                if right_right_front_speed is not None and right_right_front_speed - front_speed > self.LANE_CHANGE_SPEED_INCREASE:
                    if (surrounding_objects.right_right_back_min_distance() > 12 and surrounding_objects.right_right_front_min_distance() > 3) \
                        and (surrounding_objects.right_back_min_distance() > 12 and surrounding_objects.right_front_min_distance() > 3):
                        expect_lane_idx = current_lanes.index(self.routing_target_lane) + 1
                        if expect_lane_idx in self.available_routing_index_range:
                            return surrounding_objects.right_front_object(), surrounding_objects.right_front_min_distance(), \
                                current_lanes[expect_lane_idx], surrounding_objects.right_back_object(), surrounding_objects.right_back_min_distance(),
                if left_left_front_speed is not None and left_left_front_speed - front_speed > self.LANE_CHANGE_SPEED_INCREASE:
                    if (surrounding_objects.left_left_back_min_distance() > 12 and surrounding_objects.left_left_front_min_distance() > 3) \
                        and (surrounding_objects.left_back_min_distance() > 12 and surrounding_objects.left_front_min_distance() > 3):
                        expect_lane_idx = current_lanes.index(self.routing_target_lane) - 1
                        if expect_lane_idx in self.available_routing_index_range:
                            return surrounding_objects.left_front_object(), surrounding_objects.left_front_min_distance(), \
                                current_lanes[expect_lane_idx], surrounding_objects.left_back_object(), surrounding_objects.left_back_min_distance()

        # fall back to lane follow
        self.target_speed = self.NORMAL_SPEED
        self.overtake_timer += 1
        return surrounding_objects.front_object(), surrounding_objects.front_min_distance(), self.routing_target_lane, surrounding_objects.back_object(), surrounding_objects.back_min_distance()

    def act(self, *args, **kwargs):
        self.pdb_counter += 1
        # concat lane
        sucess = self.move_to_next_road()
        all_objects = self.control_object.lidar.get_surrounding_objects(self.control_object)

        try:
            if sucess and self.enable_lane_change:
                # perform lane change due to routing
                acc_front_obj, acc_front_dist, steering_target_lane, acc_back_obj, acc_back_dist = self.lane_change_policy(all_objects)
            else:
                # can not find routing target lane
                surrounding_objects = RuleExpertFrontBackObjects.get_find_front_back_objs(
                    all_objects,
                    self.routing_target_lane,
                    self.control_object,
                    max_distance=self.MAX_LONG_DIST,
                    target_speed=self.NORMAL_SPEED
                )
                acc_front_obj = surrounding_objects.front_object()
                acc_front_dist = surrounding_objects.front_min_distance()
                steering_target_lane = self.routing_target_lane
                acc_back_obj = surrounding_objects.back_object()
                acc_back_dist = surrounding_objects.back_min_distance()
        except Exception as e:
            # error fallback
            surrounding_objects = RuleExpertFrontBackObjects.get_find_front_back_objs(
                all_objects,
                self.routing_target_lane,
                self.control_object,
                max_distance=self.MAX_LONG_DIST,
                target_speed = self.NORMAL_SPEED
            )
            acc_front_obj = surrounding_objects.front_object()
            acc_front_dist = surrounding_objects.front_min_distance()
            steering_target_lane = self.routing_target_lane
            acc_back_obj = surrounding_objects.back_object()
            acc_back_dist = surrounding_objects.back_min_distance()
            logging.warning("IDM bug! fall back")
            print("IDM bug! fall back")
            print('except log {}'.format(e))

        # control by PID and IDM
        steering = self.steering_control(steering_target_lane)
        acc = self.acceleration_new(acc_front_obj, acc_front_dist, acc_back_obj, acc_back_dist)

        return [steering, acc]

