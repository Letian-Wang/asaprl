from metadrive.manager.agent_manager import AgentManager
from asaprl.utils.env_utils.idm_policy_utils import RuleExpertIDMPolicyHighway, RuleExpertIDMPolicyIntersection, RuleExpertIDMPolicyRoundabout
from asaprl.utils.env_utils.vehicle_utils import MacroDefaultRuleExpertVehicle
from metadrive.utils.space import ParameterSpace, VehicleParameterSpace
from metadrive.component.vehicle.vehicle_type import DefaultVehicle
from metadrive.utils import Config, safe_clip_for_small_array
from typing import Union, Dict, AnyStr, Tuple


class RuleExpertAgentManagerHighway(AgentManager):

    def _get_policy(self, obj):
        policy = RuleExpertIDMPolicyHighway(obj, self.generate_seed())
        return policy
    def before_step(self, frame = 0, wps=None):
        self._agents_finished_this_frame = dict()
        step_infos = {}
        for agent_id in self.active_agents.keys():
            policy = self.engine.get_policy(self._agent_to_object[agent_id])
            if agent_id in wps.keys():
                waypoints = wps[agent_id]
            self.get_agent(agent_id).before_macro_step(frame)
            action = policy.act(agent_id, frame, waypoints)
            step_infos[agent_id] = policy.get_action_info()
            step_infos[agent_id].update(self.get_agent(agent_id).before_step(action))
            
        finished = set()
        for v_name in self._dying_objects.keys():
            self._dying_objects[v_name][1] -= 1
            if self._dying_objects[v_name][1] == 0:  # Countdown goes to 0, it's time to remove the vehicles!
                v = self._dying_objects[v_name][0]
                self._remove_vehicle(v)
                finished.add(v_name)
        for v_name in finished:
            self._dying_objects.pop(v_name)
        return step_infos

    def _get_vehicles(self, config_dict: dict):
        from metadrive.component.vehicle.vehicle_type import random_vehicle_type, vehicle_type
        ret = {}
        v_type = MacroDefaultRuleExpertVehicle
        for agent_id, v_config in config_dict.items():
            obj = self.spawn_object(v_type, vehicle_config=v_config)
            ret[agent_id] = obj
            policy = self._get_policy(obj)
            self.engine.add_policy(obj.id, policy)
        return ret

    def after_step(self, *args, **kwargs):
        step_infos = self.for_each_active_agents(lambda v: v.after_step())

        for agent_id in self.active_agents.keys():
            ego_vehicle = self.get_agent(agent_id)
            last_position = ego_vehicle.last_macro_position
            current_position = ego_vehicle.position

            import numpy as np
            ego_vehicle.last_spd = np.sqrt(np.sum(np.square(current_position - last_position))) / ego_vehicle.physics_world_step_size

        return step_infos

class RuleExpertAgentManagerIntersection(AgentManager):

    def _get_policy(self, obj):
        policy = RuleExpertIDMPolicyIntersection(obj, self.generate_seed())
        return policy
    def before_step(self, frame = 0, wps=None):
        self._agents_finished_this_frame = dict()
        step_infos = {}
        for agent_id in self.active_agents.keys():
            policy = self.engine.get_policy(self._agent_to_object[agent_id])
            if agent_id in wps.keys():
                waypoints = wps[agent_id]
            self.get_agent(agent_id).before_macro_step(frame)
            action = policy.act(agent_id, frame, waypoints)
            step_infos[agent_id] = policy.get_action_info()
            step_infos[agent_id].update(self.get_agent(agent_id).before_step(action))
            

        finished = set()
        for v_name in self._dying_objects.keys():
            self._dying_objects[v_name][1] -= 1
            if self._dying_objects[v_name][1] == 0:  # Countdown goes to 0, it's time to remove the vehicles!
                v = self._dying_objects[v_name][0]
                self._remove_vehicle(v)
                finished.add(v_name)
        for v_name in finished:
            self._dying_objects.pop(v_name)

        return step_infos

    def after_step(self, *args, **kwargs):
        step_infos = self.for_each_active_agents(lambda v: v.after_step())

        for agent_id in self.active_agents.keys():
            ego_vehicle = self.get_agent(agent_id)
            last_position = ego_vehicle.last_macro_position
            current_position = ego_vehicle.position

            import numpy as np
            ego_vehicle.last_spd = np.sqrt(np.sum(np.square(current_position - last_position))) / ego_vehicle.physics_world_step_size

        return step_infos

    def _get_vehicles(self, config_dict: dict):
        from metadrive.component.vehicle.vehicle_type import random_vehicle_type, vehicle_type
        ret = {}
        v_type = MacroDefaultRuleExpertVehicle
        for agent_id, v_config in config_dict.items():
            obj = self.spawn_object(v_type, vehicle_config=v_config)
            ret[agent_id] = obj
            policy = self._get_policy(obj)
            self.engine.add_policy(obj.id, policy)
        return ret

class RuleExpertAgentManagerRoundabout(AgentManager):

    def _get_policy(self, obj):
        policy = RuleExpertIDMPolicyRoundabout(obj, self.generate_seed())
        return policy
    def before_step(self, frame = 0, wps=None):
        self._agents_finished_this_frame = dict()
        step_infos = {}
        for agent_id in self.active_agents.keys():
            policy = self.engine.get_policy(self._agent_to_object[agent_id])
            if agent_id in wps.keys():
                waypoints = wps[agent_id]
            self.get_agent(agent_id).before_macro_step(frame)
            action = policy.act(agent_id, frame, waypoints)
            step_infos[agent_id] = policy.get_action_info()
            step_infos[agent_id].update(self.get_agent(agent_id).before_step(action))
            
        finished = set()
        for v_name in self._dying_objects.keys():
            self._dying_objects[v_name][1] -= 1
            if self._dying_objects[v_name][1] == 0:  # Countdown goes to 0, it's time to remove the vehicles!
                v = self._dying_objects[v_name][0]
                self._remove_vehicle(v)
                finished.add(v_name)
        for v_name in finished:
            self._dying_objects.pop(v_name)

        return step_infos

    def after_step(self, *args, **kwargs):
        step_infos = self.for_each_active_agents(lambda v: v.after_step())

        for agent_id in self.active_agents.keys():
            ego_vehicle = self.get_agent(agent_id)
            last_position = ego_vehicle.last_macro_position
            current_position = ego_vehicle.position

            import numpy as np
            ego_vehicle.last_spd = np.sqrt(np.sum(np.square(current_position - last_position))) / ego_vehicle.physics_world_step_size

        return step_infos

    def _get_vehicles(self, config_dict: dict):
        from metadrive.component.vehicle.vehicle_type import random_vehicle_type, vehicle_type
        ret = {}
        v_type = MacroDefaultRuleExpertVehicle
        for agent_id, v_config in config_dict.items():
            obj = self.spawn_object(v_type, vehicle_config=v_config)
            ret[agent_id] = obj
            policy = self._get_policy(obj)
            self.engine.add_policy(obj.id, policy)
        return ret
