import copy
import logging
import math
from collections import namedtuple
from typing import Dict

from metadrive.component.lane.abs_lane import AbstractLane
from metadrive.component.map.base_map import BaseMap
from metadrive.component.road_network import Road
from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.constants import TARGET_VEHICLES, TRAFFIC_VEHICLES, OBJECT_TO_AGENT, AGENT_TO_OBJECT
from metadrive.manager.base_manager import BaseManager
from metadrive.utils import merge_dicts

from metadrive.manager.traffic_manager import TrafficManager, BlockVehicles
from asaprl.utils.env_utils.idm_policy_utils import MacroIDMPolicy

class TrafficMode:
    # Traffic vehicles will be respawn, once they arrive at the destinations
    Respawn = "respawn"

    # Traffic vehicles will be triggered only once
    Trigger = "trigger"

    # Hybrid, some vehicles are triggered once on map and disappear when arriving at destination, others exist all time
    Hybrid = "hybrid"

    # Synch, Synchronize all the vehicles immediately as well as dependent on traffic density
    Synch = "synch"

class MacroTrafficManager(TrafficManager):

    def reset(self):
        """
        Generate traffic on map, according to the mode and density
        :return: List of Traffic vehicles
        """
        map = self.current_map
        logging.debug("load scene {}".format("Use random traffic" if self.random_traffic else ""))

        # update vehicle list
        self.block_triggered_vehicles = []

        traffic_density = self.density
        if abs(traffic_density) < 1e-2:
            return
        self.respawn_lanes = self.respawn_lanes = self._get_available_respawn_lanes(map)
        if self.mode == TrafficMode.Respawn:
            # add respawn vehicle
            self._create_respawn_vehicles(map, traffic_density)
        elif self.mode == TrafficMode.Trigger or self.mode == TrafficMode.Hybrid:
            self._create_vehicles_once(map, traffic_density)
        elif self.mode == TrafficMode.Synch:
            #self._create_respawn_vehicles(map, traffic_density)
            self._create_synch_vehicles(map, traffic_density)
            self.trigger_vehicles()
        else:
            raise ValueError("No such mode named {}".format(self.mode))

        self.speed_shift_lst = [(self.generate_seed() - self.MAX_RAND_INT / 2) / self.MAX_RAND_INT * 8 for i in range(10)]
        

    def trigger_vehicles(self):
        # This will triger all static vehicles to move
        # Remember to wait untill all vehicles has fallen down
        while len(self.block_triggered_vehicles) > 0:
            block_vehicles = self.block_triggered_vehicles.pop()
            self._traffic_vehicles += block_vehicles.vehicles

    def _create_synch_vehicles(self, map: BaseMap, traffic_density: float):
        vehicle_num = 0
        for block_num, block in enumerate(map.blocks):
            xb_dist = min(abs(block.bounding_box[0]),abs(block.bounding_box[1]))
            yb_dist = min(abs(block.bounding_box[2]),abs(block.bounding_box[3]))
            dist_constrain = 200
            if xb_dist*xb_dist + yb_dist * yb_dist > dist_constrain *dist_constrain:
                continue
            trigger_lanes = block.get_intermediate_spawn_lanes()
            if self.engine.global_config["need_inverse_traffic"] and block.ID in ["S", "C", "r", "R"]:
                neg_lanes = block.block_network.get_negative_lanes()
                self.np_random.shuffle(neg_lanes)
                trigger_lanes += neg_lanes
            potential_vehicle_configs = []
            for lanes in trigger_lanes:
                for l in lanes:
                    if hasattr(self.engine, "object_manager") and l in self.engine.object_manager.accident_lanes:
                        continue
                    potential_vehicle_configs += self._propose_vehicle_configs(l)

            # How many vehicles should we spawn in this block?
            total_length = sum([lane.length for lanes in trigger_lanes for lane in lanes])
            total_spawn_points = int(math.floor(total_length / self.VEHICLE_GAP))
            total_vehicles = int(math.floor(total_spawn_points * traffic_density))
            # Generate vehicles!
            vehicles_on_block = []
            self.np_random.shuffle(potential_vehicle_configs)
            selected = potential_vehicle_configs[:min(total_vehicles, len(potential_vehicle_configs))]
            # print("We have {} candidates! We are spawning {} vehicles!".format(total_vehicles, len(selected)))

            for v_config in selected:
                if block_num == 0 and v_config['spawn_longitude'] < self.VEHICLE_GAP + self.engine.global_config["target_vehicle_configs"]['default_agent']['spawn_longitude']:
                    continue

                vehicle_type = self.random_vehicle_type()
                v_config.update(self.engine.global_config["traffic_vehicle_config"])
                random_v = self.spawn_object(vehicle_type, vehicle_config=v_config)

                self.engine.add_policy(random_v.id, MacroIDMPolicy(random_v, self.generate_seed()))
                vehicles_on_block.append(random_v)

            trigger_road = block.pre_block_socket.positive_road
            block_vehicles = BlockVehicles(trigger_road=trigger_road, vehicles=vehicles_on_block)

            self.block_triggered_vehicles.append(block_vehicles)
            vehicle_num += len(vehicles_on_block)
        self.block_triggered_vehicles.reverse()
        respawn_lanes = self._get_available_respawn_lanes(map)

    def _create_respawn_vehicles_with_density(self, traffic_density: float, lane: AbstractLane, is_respawn_lane):
        """
        Create vehicles on a lane
        :param traffic_density: traffic density according to num of vehicles per meter
        :param lane: Circular lane or Straight lane
        :param is_respawn_lane: Whether vehicles should be respawn on this lane or not
        :return: List of vehicles
        """

        _traffic_vehicles = []
        total_num = int(lane.length / self.VEHICLE_GAP)
        vehicle_longs = [i * self.VEHICLE_GAP for i in range(total_num)]
        self.np_random.shuffle(vehicle_longs)
        total_vehicles = int(math.floor(len(vehicle_longs) * traffic_density))
        total_vehicles = max(total_vehicles, 1)
        vehicle_longs = vehicle_longs[:total_vehicles]
        for long in vehicle_longs:
            vehicle_type = self.random_vehicle_type()
            traffic_v_config = {"spawn_lane_index": lane.index, "spawn_longitude": long}
            traffic_v_config.update(self.engine.global_config["traffic_vehicle_config"])
            random_v = self.spawn_object(vehicle_type, vehicle_config=traffic_v_config)
            self.engine.add_policy(random_v.id, MacroIDMPolicy(random_v, self.generate_seed()))
            _traffic_vehicles.append(random_v)
        return _traffic_vehicles