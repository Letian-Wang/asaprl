from ding.worker import SampleSerialCollector, InteractionSerialEvaluator, BaseLearner, NaiveReplayBuffer
from typing import Optional, Callable, Tuple
from collections import namedtuple
import numpy as np
import torch
import copy, pdb
import pickle, os
from ding.envs import BaseEnvManager
from ding.torch_utils import to_tensor, to_ndarray
from ding.utils import build_logger, EasyTimer, SERIAL_EVALUATOR_REGISTRY
from ding.worker import ISerialEvaluator, VectorEvalMonitor

@SERIAL_EVALUATOR_REGISTRY.register('meta-interaction')
class MetadriveEvaluator(InteractionSerialEvaluator):
    """
    Overview:
        Interaction serial evaluator class, policy interacts with env.
    Interfaces:
        __init__, reset, reset_policy, reset_env, close, should_eval, eval
    Property:
        env, policy
    """
    def eval(
            self,
            save_ckpt_fn: Callable = None,
            train_iter: int = -1,
            envstep: int = -1,
            n_episode: Optional[int] = None,
            dense_reward = False
    ) -> Tuple[bool, float]:
        '''
        Overview:
            Evaluate policy and store the best policy based on whether it reaches the highest historical reward.
        Arguments:
            - save_ckpt_fn (:obj:`Callable`): Saving ckpt function, which will be triggered by getting the best reward.
            - train_iter (:obj:`int`): Current training iteration.
            - envstep (:obj:`int`): Current env interaction step.
            - n_episode (:obj:`int`): Number of evaluation episodes.
        Returns:
            - stop_flag (:obj:`bool`): Whether this training program can be ended.
            - eval_reward (:obj:`float`): Current eval_reward.
        '''
        if n_episode is None:
            n_episode = self._default_n_episode
        assert n_episode is not None, "please indicate eval n_episode"
        envstep_count = 0
        eval_monitor = VectorEvalMonitor(self._env.env_num, n_episode)
        self._env.reset()
        self._policy.reset()

        z_success_times = 0
        z_fail_times = 0
        overtake_vehicle_num = 0
        collision_num = 0
        out_of_time_num = 0
        complete_ratio_list = []
        episode_velocity_lst = [[] for i in range(n_episode)]
        episode_skill_horizon_lst = [[] for i in range(n_episode)]
        episode_reward_total = [0 for i in range(n_episode)]
        episode_reward_pass_car = [0 for i in range(n_episode)]
        episode_reward_on_broken_lane = [0 for i in range(n_episode)]
        episode_reward_out_of_road = [0 for i in range(n_episode)]
        episode_reward_crash = [0 for i in range(n_episode)]
        episode_reward_arrive_desti = [0 for i in range(n_episode)]
        episode_reward_out_of_time = [0 for i in range(n_episode)]
        episode_reward_complete_road = [0 for i in range(n_episode)]

        with self._timer:
            while not eval_monitor.is_finished():
                obs = self._env.ready_obs
                obs = to_tensor(obs, dtype=torch.float32)
                policy_output = self._policy.forward(obs)
                actions = {i: a['action'] for i, a in policy_output.items()}
                actions = to_ndarray(actions)
                timesteps = self._env.step(actions)
                timesteps = to_tensor(timesteps, dtype=torch.float32)
                for env_id, t in timesteps.items():
                    current_episode_index = eval_monitor.get_current_episode() - 1
                    episode_velocity_lst[current_episode_index].append(t.info['vehicle_last_speed'].item())
                    episode_skill_horizon_lst[current_episode_index].append(t.info['skill_horizon'].item())
                    if not dense_reward:        # reward log
                        episode_reward_pass_car[current_episode_index] += t.info['step_reward_pass_car']
                        episode_reward_on_broken_lane[current_episode_index] += t.info['step_reward_on_broken_lane']
                        episode_reward_out_of_road[current_episode_index] += t.info['step_reward_out_of_road']
                        episode_reward_crash[current_episode_index] += t.info['step_reward_crash']
                        episode_reward_arrive_desti[current_episode_index] += t.info['step_reward_arrive_desti']
                        episode_reward_out_of_time[current_episode_index] += t.info['step_reward_out_of_time']
                        episode_reward_complete_road[current_episode_index] += t.info['step_reward_complete_road']
                        episode_reward_total[current_episode_index] += t.info['step_reward']
                    if t.info.get('abnormal', False):
                        # If there is an abnormal timestep, reset all the related variables(including this env).
                        self._policy.reset([env_id])
                        continue
                    if t.done:
                        # Env reset is done by env_manager automatically.
                        self._policy.reset([env_id])
                        reward = t.info['final_eval_reward']
                        arrive_dest = t.info['arrive_dest']
                        overtake_vehicle_num += t.info['overtake_vehicle_num']
                        if t.info['crash_vehicle'] or t.info['crash_object'] or t.info['crash_building'] or t.info['crash'] or t.info['out_of_road']:
                            collision_num += 1
                        if t.info['max_step']:
                            out_of_time_num += 1
                        if arrive_dest:
                            z_success_times += 1
                        else:
                            z_fail_times += 1
                        if 'complete_ratio' in t.info:
                            complete_ratio_list.append(float(t.info['complete_ratio']))
                        if 'episode_info' in t.info:
                            eval_monitor.update_info(env_id, t.info['episode_info'])
                        eval_monitor.update_reward(env_id, reward)
                        self._logger.info(
                            "[EVALUATOR]env {} finish episode, final reward: {}, current episode: {}".format(
                                env_id, eval_monitor.get_latest_reward(env_id), eval_monitor.get_current_episode()
                            )
                        )
                    envstep_count += int(t[-1]['envstep'])
                    
        duration = self._timer.value
        episode_reward = eval_monitor.get_episode_reward()
        info = {
            'train_iter': train_iter,
            'ckpt_name': 'iteration_{}.pth.tar'.format(train_iter),
            'episode_count': n_episode,
            'envstep_count': envstep_count,
            'avg_envstep_per_episode': envstep_count / n_episode,
            'evaluate_time': duration,
            'avg_envstep_per_sec': envstep_count / duration,
            'avg_time_per_episode': n_episode / duration,
            'reward_mean': np.mean(episode_reward),
            'reward_std': np.std(episode_reward),
            'reward_max': np.max(episode_reward),
            'reward_min': np.min(episode_reward),
        }
        episode_info = eval_monitor.get_episode_info()
        if episode_info is not None:
            info.update(episode_info)
        self._logger.info(self._logger.get_tabulate_vars_hor(info))
        for k, v in info.items():
            if k in ['train_iter', 'ckpt_name', 'each_reward']:
                continue
            if not np.isscalar(v):
                continue
            self._tb_logger.add_scalar('{}_iter/'.format(self._instance_name) + k, v, train_iter)
            self._tb_logger.add_scalar('{}_step/'.format(self._instance_name) + k, v, envstep)
        eval_reward = np.mean(episode_reward)
        if eval_reward > self._max_eval_reward:
            if save_ckpt_fn:
                save_ckpt_fn('ckpt_best_iter{}_step{}.pth.tar'.format(train_iter, envstep))
            self._max_eval_reward = eval_reward
        stop_flag = eval_reward >= self._stop_value and train_iter > 0
        if stop_flag:
            self._logger.info(
                "[DI-engine serial pipeline] " +
                "Current eval_reward: {} is greater than stop_value: {}".format(eval_reward, self._stop_value) +
                ", so your RL agent is converged, you can refer to 'log/evaluator/evaluator_logger.txt' for details."
            )
        
        # some printings
        def cal_mean_of_nested_list(lst):
            num = 0 
            sum = 0
            for one_list in lst:
                for item in one_list:
                    num += 1
                    sum += item
            return sum/num
        total_episode = float(z_success_times + z_fail_times)
        success_ratio = float(z_success_times) / total_episode
        collision_rate = float(collision_num) / total_episode
        out_of_time_rate = float(out_of_time_num) / total_episode
        overtake_vehicle_num_per_episode = float(overtake_vehicle_num) / total_episode
        print('evaluator log')
        print('success times: {}'.format(z_success_times))
        print('fail_times: {}'.format(z_fail_times))
        print('success ratio: {}'.format(success_ratio))
        print('collision_rate: {}'.format(collision_rate))
        print('out_of_time_rate: {}'.format(out_of_time_rate))
        print('overtake_vehicle_num_per_episode: {}'.format(overtake_vehicle_num_per_episode))
        print('road_completion_ratio: {}'.format(np.mean(complete_ratio_list)))
        print('average_velocity: {}'.format(cal_mean_of_nested_list(episode_velocity_lst)))
        
        self._tb_logger.add_scalar('episode_info/succ_rate_iter', success_ratio, train_iter)
        self._tb_logger.add_scalar('episode_info/succ_rate_step', success_ratio, envstep)
        self._tb_logger.add_scalar('episode_info/skill_horizon_iter', cal_mean_of_nested_list(episode_skill_horizon_lst), train_iter)
        self._tb_logger.add_scalar('episode_info/skill_horizon_step', cal_mean_of_nested_list(episode_skill_horizon_lst), envstep)
        self._tb_logger.add_scalar('episode_info/velocity_iter', cal_mean_of_nested_list(episode_velocity_lst), train_iter)
        self._tb_logger.add_scalar('episode_info/velocity_step', cal_mean_of_nested_list(episode_velocity_lst), envstep)
        self._tb_logger.add_scalar('episode_info/complete_ratio_mean_iter', np.mean(complete_ratio_list), train_iter)
        self._tb_logger.add_scalar('episode_info/complete_ratio_mean_step', np.mean(complete_ratio_list), envstep)
        self._tb_logger.add_scalar('episode_info/complete_ratio_std_iter', np.std(complete_ratio_list), train_iter)
        self._tb_logger.add_scalar('episode_info/complete_ratio_std_step', np.std(complete_ratio_list), envstep)
        self._tb_logger.add_scalar('episode_info/collision_rate_iter', collision_rate, train_iter)
        self._tb_logger.add_scalar('episode_info/collision_rate_step', collision_rate, envstep)
        self._tb_logger.add_scalar('episode_info/out_of_time_rate_iter', out_of_time_rate, train_iter)
        self._tb_logger.add_scalar('episode_info/out_of_time_rate_step', out_of_time_rate, envstep)
        self._tb_logger.add_scalar('episode_info/pass_car_per_episode_iter', overtake_vehicle_num_per_episode, train_iter)
        self._tb_logger.add_scalar('episode_info/pass_car_per_episode_step', overtake_vehicle_num_per_episode, envstep)

        self._tb_logger.add_scalar('episode_reward_info_iter/total', np.mean(episode_reward_total), train_iter)
        self._tb_logger.add_scalar('episode_reward_info_step/total', np.mean(episode_reward_total), envstep)
        self._tb_logger.add_scalar('episode_reward_info_iter/pass_car', np.mean(episode_reward_pass_car), train_iter)
        self._tb_logger.add_scalar('episode_reward_info_step/pass_car', np.mean(episode_reward_pass_car), envstep)
        self._tb_logger.add_scalar('episode_reward_info_iter/on_broken_lane', np.mean(episode_reward_on_broken_lane), train_iter)
        self._tb_logger.add_scalar('episode_reward_info_step/on_broken_lane', np.mean(episode_reward_on_broken_lane), envstep)
        self._tb_logger.add_scalar('episode_reward_info_iter/out_of_road', np.mean(episode_reward_out_of_road), train_iter)
        self._tb_logger.add_scalar('episode_reward_info_step/out_of_road', np.mean(episode_reward_out_of_road), envstep)
        self._tb_logger.add_scalar('episode_reward_info_iter/crash', np.mean(episode_reward_crash), train_iter)
        self._tb_logger.add_scalar('episode_reward_info_step/crash', np.mean(episode_reward_crash), envstep)
        self._tb_logger.add_scalar('episode_reward_info_iter/arrive_desti', np.mean(episode_reward_arrive_desti), train_iter)
        self._tb_logger.add_scalar('episode_reward_info_step/arrive_desti', np.mean(episode_reward_arrive_desti), envstep)
        self._tb_logger.add_scalar('episode_reward_info_iter/out_of_time', np.mean(episode_reward_out_of_time), train_iter)
        self._tb_logger.add_scalar('episode_reward_info_step/out_of_time', np.mean(episode_reward_out_of_time), envstep)
        self._tb_logger.add_scalar('episode_reward_info_iter/complete_road', np.mean(episode_reward_complete_road), train_iter)
        self._tb_logger.add_scalar('episode_reward_info_step/complete_road', np.mean(episode_reward_complete_road), envstep)

        return stop_flag, eval_reward

@SERIAL_EVALUATOR_REGISTRY.register('meta-expert-interaction')
class MetadriveExpertEvaluator(InteractionSerialEvaluator):
    """
    Overview:
        Interaction serial evaluator class, policy interacts with env.
    Interfaces:
        __init__, reset, reset_policy, reset_env, close, should_eval, eval
    Property:
        env, policy
    """
    def eval(
            self,
            scenario = 'highway',
            save_ckpt_fn: Callable = None,
            train_iter: int = -1,
            envstep: int = -1,
            n_episode: Optional[int] = None,
            dense_reward = False
    ) -> Tuple[bool, float]:
        '''
        Overview:
            Evaluate policy and store the best policy based on whether it reaches the highest historical reward.
        Arguments:
            - save_ckpt_fn (:obj:`Callable`): Saving ckpt function, which will be triggered by getting the best reward.
            - train_iter (:obj:`int`): Current training iteration.
            - envstep (:obj:`int`): Current env interaction step.
            - n_episode (:obj:`int`): Number of evaluation episodes.
        Returns:
            - stop_flag (:obj:`bool`): Whether this training program can be ended.
            - eval_reward (:obj:`float`): Current eval_reward.
        '''
        if n_episode is None:
            n_episode = self._default_n_episode
        assert n_episode is not None, "please indicate eval n_episode"
        envstep_count = 0
        eval_monitor = VectorEvalMonitor(self._env.env_num, n_episode)
        self._env.reset()
        self._policy.reset()

        z_success_times = 0
        z_fail_times = 0
        overtake_vehicle_num = 0
        collision_num = 0
        out_of_time_num = 0 
        out_of_road_num = 0 
        complete_ratio_list = []
        episode_velocity_lst = [[] for i in range(n_episode)]
        episode_skill_horizon_lst = [[] for i in range(n_episode)]
        episode_reward_total = [0 for i in range(n_episode)]
        episode_reward_pass_car = [0 for i in range(n_episode)]
        episode_reward_on_broken_lane = [0 for i in range(n_episode)]
        episode_reward_out_of_road = [0 for i in range(n_episode)]
        episode_reward_crash = [0 for i in range(n_episode)]
        episode_reward_arrive_desti = [0 for i in range(n_episode)]
        episode_reward_out_of_time = [0 for i in range(n_episode)]
        episode_reward_complete_road = [0 for i in range(n_episode)]

        if not os.path.exists('./demonstration_RL_expert') :
            os.mkdir('./demonstration_RL_expert')
        if not os.path.exists('./demonstration_RL_expert/{}'.format(scenario)) :
            os.mkdir('./demonstration_RL_expert/{}'.format(scenario))
        exist_demo_num = len(os.listdir('./demonstration_RL_expert/{}/'.format(scenario)))
        demon_data_lst = [{'obs':[], 'abs_state':[], 'rela_state':[], 'latent_var':[], 'logit': [], 'current_spd':[]} for i in range(n_episode)]

        with self._timer:
            while not eval_monitor.is_finished():
                obs = self._env.ready_obs
                obs = to_tensor(obs, dtype=torch.float32)
                policy_output = self._policy.forward(obs)
                actions = {i: a['action'] for i, a in policy_output.items()}
                logits = {i: a['logit'] for i, a in policy_output.items()}
                actions = to_ndarray(actions)
                timesteps = self._env.step(actions)
                timesteps = to_tensor(timesteps, dtype=torch.float32)
                for env_id, t in timesteps.items():
                    current_episode_index = eval_monitor.get_current_episode() - 1
                    episode_velocity_lst[current_episode_index].append(t.info['vehicle_last_speed'].item())
                    episode_skill_horizon_lst[current_episode_index].append(t.info['skill_horizon'].item())
                    if not dense_reward:    # reward log
                        episode_reward_pass_car[current_episode_index] += t.info['step_reward_pass_car']
                        episode_reward_on_broken_lane[current_episode_index] += t.info['step_reward_on_broken_lane']
                        episode_reward_out_of_road[current_episode_index] += t.info['step_reward_out_of_road']
                        episode_reward_crash[current_episode_index] += t.info['step_reward_crash']
                        episode_reward_arrive_desti[current_episode_index] += t.info['step_reward_arrive_desti']
                        episode_reward_out_of_time[current_episode_index] += t.info['step_reward_out_of_time']
                        episode_reward_complete_road[current_episode_index] += t.info['step_reward_complete_road']
                        episode_reward_total[current_episode_index] += t.info['step_reward']
                    # collect data
                    demon_data_lst[current_episode_index]['obs'].extend(to_ndarray(t.info['obs_one_skill']))
                    demon_data_lst[current_episode_index]['abs_state'].extend((to_ndarray(t.info['abs_state_one_skill'])))
                    demon_data_lst[current_episode_index]['rela_state'].append((to_ndarray(t.info['relative_state_one_skill'])))
                    demon_data_lst[current_episode_index]['current_spd'].append(t.info['vehicle_last_speed'])
                    demon_data_lst[current_episode_index]['latent_var'].append(actions[env_id])
                    demon_data_lst[current_episode_index]['logit'].append(logits[env_id])
                    if t.info.get('abnormal', False):
                        # If there is an abnormal timestep, reset all the related variables(including this env).
                        self._policy.reset([env_id])
                        continue
                    if t.done:
                        # Env reset is done by env_manager automatically.
                        self._policy.reset([env_id])
                        reward = t.info['final_eval_reward']
                        arrive_dest = t.info['arrive_dest']
                        overtake_vehicle_num += t.info['overtake_vehicle_num']
                        if (t.info['crash_vehicle'] or t.info['crash_object'] or t.info['crash_building'] or t.info['crash'] or t.info['out_of_road']):
                            collision_num += 1
                            if t.info['out_of_road']:
                                out_of_road_num += 1
                        if t.info['max_step']:
                            out_of_time_num += 1
                        if t.info['arrive_dest']:
                            z_success_times += 1
                            # only store the data when the episode is successful
                            with open('./demonstration_RL_expert/{}/{}_expert_data_{}.pickle'.format(scenario, scenario, z_success_times + exist_demo_num), 'wb') as handle:
                                pickle.dump(demon_data_lst[current_episode_index], handle, protocol=pickle.HIGHEST_PROTOCOL)
                        else:
                            z_fail_times += 1
                        if 'complete_ratio' in t.info:
                            complete_ratio_list.append(float(t.info['complete_ratio']))
                        if 'episode_info' in t.info:
                            eval_monitor.update_info(env_id, t.info['episode_info'])
                        eval_monitor.update_reward(env_id, reward)
                        self._logger.info(
                            "[EVALUATOR]env {} finish episode, final reward: {}, current episode: {}".format(
                                env_id, eval_monitor.get_latest_reward(env_id), eval_monitor.get_current_episode()
                            )
                        )
                        print("collecting {} env, success {}, failure {}".format(z_success_times+z_fail_times, z_success_times, z_fail_times))
                        demon_data_lst[current_episode_index] = {'obs':[], 'abs_state':[], 'rela_state':[], 'action':[], 'logit': []}
                    envstep_count += int(t[-1]['envstep'])
                    
        duration = self._timer.value
        episode_reward = eval_monitor.get_episode_reward()
        info = {
            'train_iter': train_iter,
            'ckpt_name': 'iteration_{}.pth.tar'.format(train_iter),
            'episode_count': n_episode,
            'envstep_count': envstep_count,
            'avg_envstep_per_episode': envstep_count / n_episode,
            'evaluate_time': duration,
            'avg_envstep_per_sec': envstep_count / duration,
            'avg_time_per_episode': n_episode / duration,
            'reward_mean': np.mean(episode_reward),
            'reward_std': np.std(episode_reward),
            'reward_max': np.max(episode_reward),
            'reward_min': np.min(episode_reward),
        }
        episode_info = eval_monitor.get_episode_info()
        if episode_info is not None:
            info.update(episode_info)
        self._logger.info(self._logger.get_tabulate_vars_hor(info))
        eval_reward = np.mean(episode_reward)

        # some printings
        def cal_mean_of_nested_list(lst):
            num = 0 
            sum = 0
            for one_list in lst:
                for item in one_list:
                    num += 1
                    sum += item
            return sum/num
        total_episode = float(z_success_times + z_fail_times)
        success_ratio = float(z_success_times) / total_episode
        collision_rate = float(collision_num) / total_episode
        out_of_time_rate = float(out_of_time_num) / total_episode
        overtake_vehicle_num_per_episode = float(overtake_vehicle_num) / total_episode
        out_of_road_rate = float(out_of_road_num) / total_episode
        print('evaluator log')
        print('success times: {}'.format(z_success_times))
        print('fail_times: {}'.format(z_fail_times))
        print('success ratio: {}'.format(success_ratio))
        print('collision_rate: {}'.format(collision_rate))
        print('out_of_road_rate: {}'.format(out_of_road_rate))
        print('out_of_time_rate: {}'.format(out_of_time_rate))
        print('overtake_vehicle_num_per_episode: {}'.format(overtake_vehicle_num_per_episode))
        print('road_completion_ratio: {}'.format(np.mean(complete_ratio_list)))
        print('average_velocity: {}'.format(cal_mean_of_nested_list(episode_velocity_lst)))
        
        return eval_reward

@SERIAL_EVALUATOR_REGISTRY.register('meta-ruleexpert-interaction')
class MetadriveRuleExpertEvaluator(InteractionSerialEvaluator):
    """
    Overview:
        Interaction serial evaluator class, policy interacts with env.
    Interfaces:
        __init__, reset, reset_policy, reset_env, close, should_eval, eval
    Property:
        env, policy
    """
    def eval(
            self,
            scenario = 'highway',
            save_ckpt_fn: Callable = None,
            train_iter: int = -1,
            envstep: int = -1,
            n_episode: Optional[int] = None,
            dense_reward = False
    ) -> Tuple[bool, float]:
        '''
        Overview:
            Evaluate policy and store the best policy based on whether it reaches the highest historical reward.
        Arguments:
            - save_ckpt_fn (:obj:`Callable`): Saving ckpt function, which will be triggered by getting the best reward.
            - train_iter (:obj:`int`): Current training iteration.
            - envstep (:obj:`int`): Current env interaction step.
            - n_episode (:obj:`int`): Number of evaluation episodes.
        Returns:
            - stop_flag (:obj:`bool`): Whether this training program can be ended.
            - eval_reward (:obj:`float`): Current eval_reward.
        '''
        if n_episode is None:
            n_episode = self._default_n_episode
        assert n_episode is not None, "please indicate eval n_episode"
        envstep_count = 0
        eval_monitor = VectorEvalMonitor(self._env.env_num, n_episode)
        self._env.reset()
        self._policy.reset()

        z_success_times = 0
        z_fail_times = 0
        overtake_vehicle_num = 0
        collision_num = 0
        out_of_time_num = 0
        out_of_road_num = 0 
        complete_ratio_list = []
        episode_velocity_lst = [[] for i in range(n_episode)]
        episode_skill_horizon_lst = [[] for i in range(n_episode)]
        episode_reward_total = [0 for i in range(n_episode)]
        episode_reward_pass_car = [0 for i in range(n_episode)]
        episode_reward_on_broken_lane = [0 for i in range(n_episode)]
        episode_reward_out_of_road = [0 for i in range(n_episode)]
        episode_reward_crash = [0 for i in range(n_episode)]
        episode_reward_arrive_desti = [0 for i in range(n_episode)]
        episode_reward_out_of_time = [0 for i in range(n_episode)]
        episode_reward_complete_road = [0 for i in range(n_episode)]

        if not os.path.exists('./demonstration_rule_expert') :
            os.mkdir('./demonstration_rule_expert')
        if not os.path.exists('./demonstration_rule_expert/{}'.format(scenario)) :
            os.mkdir('./demonstration_rule_expert/{}'.format(scenario))
        exist_demo_num = len(os.listdir('./demonstration_rule_expert/{}/'.format(scenario)))
        demon_data_lst = [{'obs':[], 'abs_state':[], 'rela_state':[], 'action':[], 'reward':[], 'vehicle_start_speed':[]} for i in range(n_episode)]

        with self._timer:
            while not eval_monitor.is_finished():
                obs = self._env.ready_obs
                obs = to_tensor(obs, dtype=torch.float32)
                policy_output = self._policy.forward(obs)
                actions = {i: a['action'] for i, a in policy_output.items()}
                actions = to_ndarray(actions)
                timesteps = self._env.step(actions)
                timesteps = to_tensor(timesteps, dtype=torch.float32)
                for env_id, t in timesteps.items():
                    current_episode_index = eval_monitor.get_current_episode() - 1
                    episode_velocity_lst[current_episode_index].append(t.info['vehicle_last_speed'].item())
                    episode_skill_horizon_lst[current_episode_index].append(t.info['skill_horizon'].item())
                    if not dense_reward:    # reward log
                        episode_reward_pass_car[current_episode_index] += t.info['step_reward_pass_car']
                        episode_reward_on_broken_lane[current_episode_index] += t.info['step_reward_on_broken_lane']
                        episode_reward_out_of_road[current_episode_index] += t.info['step_reward_out_of_road']
                        episode_reward_crash[current_episode_index] += t.info['step_reward_crash']
                        episode_reward_arrive_desti[current_episode_index] += t.info['step_reward_arrive_desti']
                        episode_reward_out_of_time[current_episode_index] += t.info['step_reward_out_of_time']
                        episode_reward_complete_road[current_episode_index] += t.info['step_reward_complete_road']
                        episode_reward_total[current_episode_index] += t.info['step_reward']
                    # collect data
                    demon_data_lst[current_episode_index]['obs'].extend(to_ndarray(t.info['obs_one_skill']))
                    demon_data_lst[current_episode_index]['abs_state'].extend((to_ndarray(t.info['abs_state_one_skill'])))
                    demon_data_lst[current_episode_index]['rela_state'].append((to_ndarray(t.info['relative_state_one_skill'])))
                    demon_data_lst[current_episode_index]['action'].append(actions[env_id])
                    demon_data_lst[current_episode_index]['reward'].append(t.info['step_reward'].item())
                    demon_data_lst[current_episode_index]['vehicle_start_speed'].append(t.info['vehicle_start_speed'].item())
                    if t.info.get('abnormal', False):
                        # If there is an abnormal timestep, reset all the related variables(including this env).
                        self._policy.reset([env_id])
                        continue
                    if t.done:
                        # Env reset is done by env_manager automatically.
                        self._policy.reset([env_id])
                        reward = t.info['final_eval_reward']
                        arrive_dest = t.info['arrive_dest']
                        overtake_vehicle_num += t.info['overtake_vehicle_num']
                        if (t.info['crash_vehicle'] or t.info['crash_object'] or t.info['crash_building'] or t.info['crash'] or t.info['out_of_road']):
                            collision_num += 1
                            if t.info['out_of_road']:
                                out_of_road_num += 1
                        if t.info['max_step']:
                            out_of_time_num += 1
                        if t.info['arrive_dest']:
                            z_success_times += 1
                            # only store the data when the episode is successful
                            with open('./demonstration_rule_expert/{}/{}_expert_data_{}.pickle'.format(scenario, scenario, z_success_times + exist_demo_num), 'wb') as handle:
                                pickle.dump(demon_data_lst[current_episode_index], handle, protocol=pickle.HIGHEST_PROTOCOL)
                        else:
                            z_fail_times += 1
                        if 'complete_ratio' in t.info:
                            complete_ratio_list.append(float(t.info['complete_ratio']))
                        if 'episode_info' in t.info:
                            eval_monitor.update_info(env_id, t.info['episode_info'])
                        eval_monitor.update_reward(env_id, reward)
                        self._logger.info(
                            "[EVALUATOR]env {} finish episode, final reward: {}, current episode: {}".format(
                                env_id, eval_monitor.get_latest_reward(env_id), eval_monitor.get_current_episode()
                            )
                        )
                        print("collecting {} env, success {}, failure {}".format(z_success_times+z_fail_times, z_success_times, z_fail_times))
                        demon_data_lst[current_episode_index] = {'obs':[], 'abs_state':[], 'rela_state':[], 'action':[], 'reward':[], 'vehicle_start_speed':[]}
                    envstep_count += int(t[-1]['envstep'])
                    
        duration = self._timer.value
        episode_reward = eval_monitor.get_episode_reward()
        info = {
            'train_iter': train_iter,
            'ckpt_name': 'iteration_{}.pth.tar'.format(train_iter),
            'episode_count': n_episode,
            'envstep_count': envstep_count,
            'avg_envstep_per_episode': envstep_count / n_episode,
            'evaluate_time': duration,
            'avg_envstep_per_sec': envstep_count / duration,
            'avg_time_per_episode': n_episode / duration,
            'reward_mean': np.mean(episode_reward),
            'reward_std': np.std(episode_reward),
            'reward_max': np.max(episode_reward),
            'reward_min': np.min(episode_reward),
        }
        episode_info = eval_monitor.get_episode_info()
        if episode_info is not None:
            info.update(episode_info)
        self._logger.info(self._logger.get_tabulate_vars_hor(info))
        eval_reward = np.mean(episode_reward)

        # some printings
        def cal_mean_of_nested_list(lst):
            num = 0 
            sum = 0
            for one_list in lst:
                for item in one_list:
                    num += 1
                    sum += item
            return sum/num
        total_episode = float(z_success_times + z_fail_times)
        success_ratio = float(z_success_times) / total_episode
        collision_rate = float(collision_num) / total_episode
        out_of_time_rate = float(out_of_time_num) / total_episode
        overtake_vehicle_num_per_episode = float(overtake_vehicle_num) / total_episode
        out_of_road_rate = float(out_of_road_num) / total_episode
        print('evaluator log')
        print('success times: {}'.format(z_success_times))
        print('fail_times: {}'.format(z_fail_times))
        print('success ratio: {}'.format(success_ratio))
        print('collision_rate: {}'.format(collision_rate))
        print('out_of_road_rate: {}'.format(out_of_road_rate))
        print('out_of_time_rate: {}'.format(out_of_time_rate))
        print('overtake_vehicle_num_per_episode: {}'.format(overtake_vehicle_num_per_episode))
        print('road_completion_ratio: {}'.format(np.mean(complete_ratio_list)))
        print('average_velocity: {}'.format(cal_mean_of_nested_list(episode_velocity_lst)))

        return eval_reward

def clean_old_model(exp_name):
    from os import listdir, remove
    from os.path import isfile, join
    ckpt_path = "./{}/ckpt/".format(exp_name)
    allfiles = [join(ckpt_path, f) for f in listdir(ckpt_path) if isfile(join(ckpt_path, f))]
    largest_envstep = 0
    for file in allfiles:
        if "best" in file:
            print(file)
            print(file[file.index('step') + 4:file.index('.pth')])
            envstep = int(file[file.index('step') + 4:file.index('.pth')])
            if envstep > largest_envstep:
                largest_envstep = envstep
    for file in allfiles:
        if "best" in file:
            envstep = int(file[file.index('step') + 4:file.index('.pth')])
            if envstep != largest_envstep:
                remove(file)