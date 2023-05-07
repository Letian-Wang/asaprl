from typing import List, Dict, Any, Tuple, Union
from collections import namedtuple
import copy, pdb
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal, Independent

from ding.torch_utils import to_device
from ding.rl_utils import v_1step_td_data, v_1step_td_error
from ding.utils import POLICY_REGISTRY
from ding.policy.sac import SACPolicy
from ding.policy.common_utils import default_preprocess_learn
from ding.utils import POLICY_REGISTRY
from asaprl.policy.prior_network import get_prior_network
from typing import Optional, List, Dict, Any, Tuple, Union


@POLICY_REGISTRY.register('prior_sac')
class PriorSAC(SACPolicy):

    def __init__(
            self,
            cfg: dict,
            model: Optional[Union[type, torch.nn.Module]] = None,
            enable_field: Optional[List[str]] = None
    ) -> None:
        super().__init__(cfg, model, enable_field)
        self.prior_network = get_prior_network(self._cfg.ACTOR_PRIOR_LOAD_DIR, action_shape=self._cfg.model.action_shape)
        if self._cuda:
            self.prior_network = to_device(self.prior_network, self._device)

    def reward_from_prior_network(self, observations, actions):

        ''' prior reward '''
        expert_action = self.prior_network.forward(observations, 'compute_actor')
        expert_action_mu = expert_action['logit'][0]
        expert_latent_action = torch.tanh(expert_action_mu).cpu().detach().numpy()
        
        executed_action = np.expand_dims(actions.cpu().detach().numpy(),0)
        expert_reward = np.mean(np.exp(-np.abs(expert_latent_action - executed_action)))

        return expert_reward
    
    def _forward_learn(self, data: dict, train_iter) -> Dict[str, Any]:
        """
        Overview:
            Forward and backward function of learn mode.
        Arguments:
            - data (:obj:`dict`): Dict type data, including at least ['obs', 'action', 'reward', 'next_obs']
        Returns:
            - info_dict (:obj:`Dict[str, Any]`): Including current lr, loss, target_q_value and other \
                running information.
        """
        loss_dict = {}
        data = default_preprocess_learn(
            data,
            use_priority=self._priority,
            use_priority_IS_weight=self._cfg.priority_IS_weight,
            ignore_done=self._cfg.learn.ignore_done,
            use_nstep=False
        )
        if self._cuda:
            data = to_device(data, self._device)

        self._learn_model.train()
        self._target_model.train()
        obs = data['obs']
        next_obs = data['next_obs']
        reward = data['reward']
        done = data['done']
        actions = data['action']
        proir_num = 0
        all_num = 0

        if self._cfg.reward_augment and not (self._cfg.iter_turnoff_prior != 0 and train_iter > self._cfg.iter_turnoff_prior):
            reward += self.reward_from_prior_network(obs, actions)

        # 1. predict q value
        q_value = self._learn_model.forward(data, mode='compute_critic')['q_value']

        # 2. predict target value depend self._value_network.
        if self._value_network:
            v_value = self._learn_model.forward(obs, mode='compute_value_critic')['v_value']
            with torch.no_grad():
                next_v_value = self._target_model.forward(next_obs, mode='compute_value_critic')['v_value']
            target_q_value = next_v_value
        else:
            # target q value.
            with torch.no_grad():
                (mu, sigma) = self._learn_model.forward(next_obs, mode='compute_actor')['logit']

                dist = Independent(Normal(mu, sigma), 1)
                pred = dist.rsample()
                next_action = torch.tanh(pred)
                y = 1 - next_action.pow(2) + 1e-6
                # keep dimension for loss computation (usually for action space is 1 env. e.g. pendulum)
                next_log_prob = dist.log_prob(pred).unsqueeze(-1)
                next_log_prob = next_log_prob - torch.log(y).sum(-1, keepdim=True)

                # use the Kl divergence between actor and prior network to replace the original entropy term 
                if self._cfg.prior_kl and not (self._cfg.iter_turnoff_prior != 0 and train_iter > self._cfg.iter_turnoff_prior):
                    no_prior_next_log_prob = next_log_prob

                    (prior_mu, prior_sigma) = self.prior_network.forward(next_obs, 'compute_actor')['logit']
                    prior_dist = Independent(Normal(prior_mu, prior_sigma), 1)
                    # keep dimension for loss computation (usually for action space is 1 env. e.g. pendulum)
                    prior_next_log_prob = prior_dist.log_prob(pred).unsqueeze(-1)
                    prior_next_log_prob = prior_next_log_prob - torch.log(y).sum(-1, keepdim=True)

                    next_log_prob = no_prior_next_log_prob - prior_next_log_prob

                # selective Kl divergence, determined by which q value is higher between the prior network and the RL actor 
                if self._cfg.prior_filter_kl and not (self._cfg.iter_turnoff_prior != 0 and train_iter > self._cfg.iter_turnoff_prior):
                    no_prior_next_log_prob = next_log_prob

                    (prior_mu, prior_sigma) = self.prior_network.forward(next_obs, 'compute_actor')['logit']
                    prior_dist = Independent(Normal(prior_mu, prior_sigma), 1)
                    prior_next_log_prob = prior_dist.log_prob(pred).unsqueeze(-1)
                    prior_next_log_prob = prior_next_log_prob - torch.log(y).sum(-1, keepdim=True)
                    prior_next_log_prob = no_prior_next_log_prob - prior_next_log_prob

                    # the Q value of the action from the prior network
                    (prior_mu, prior_sigma) = self.prior_network.forward(data['obs'], 'compute_actor')['logit']
                    prior_dist = Independent(Normal(prior_mu, prior_sigma), 1)
                    prior_pred = prior_dist.rsample()
                    prior_action = torch.tanh(prior_pred)
                    temp_prior_data = {'obs': obs, 'action': prior_action}
                    prior_q_value = self._learn_model.forward(temp_prior_data, mode='compute_critic')['q_value']
                    prior_q_value_min = torch.unsqueeze(torch.min(torch.vstack((prior_q_value[0], prior_q_value[1])), 0)[0], 1)

                    # the Q value of the action from the RL actor network
                    (mu, sigma) = self._learn_model.forward(data['obs'], mode='compute_actor')['logit']
                    dist = Independent(Normal(mu, sigma), 1)
                    pred = dist.rsample()
                    action = torch.tanh(pred)
                    temp_actor_data = {'obs': obs, 'action': action}
                    actor_q_value = self._learn_model.forward(temp_actor_data, mode='compute_critic')['q_value']
                    actor_q_value_min = torch.unsqueeze(torch.min(torch.vstack((actor_q_value[0], actor_q_value[1])), 0)[0], 1)

                    # selective KL
                    next_log_prob = torch.where(prior_q_value_min > actor_q_value_min, prior_next_log_prob, no_prior_next_log_prob)

                    # some counters
                    proir_num += torch.where(prior_q_value_min > actor_q_value_min)[0].shape[0]
                    all_num += prior_q_value_min.shape[0]

                next_data = {'obs': next_obs, 'action': next_action}
                target_q_value = self._target_model.forward(next_data, mode='compute_critic')['q_value']
                # the value of a policy according to the maximum entropy objective
                if self._twin_critic:
                    # find min one as target q value
                    target_q_value = torch.min(target_q_value[0],
                                               target_q_value[1]) - self._alpha * next_log_prob.squeeze(-1)
                else:
                    target_q_value = target_q_value - self._alpha * next_log_prob.squeeze(-1)

        # 3. compute q loss
        if self._twin_critic:
            q_data0 = v_1step_td_data(q_value[0], target_q_value, reward, done, data['weight'])
            loss_dict['critic_loss'], td_error_per_sample0 = v_1step_td_error(q_data0, self._gamma)
            q_data1 = v_1step_td_data(q_value[1], target_q_value, reward, done, data['weight'])
            loss_dict['twin_critic_loss'], td_error_per_sample1 = v_1step_td_error(q_data1, self._gamma)
            td_error_per_sample = (td_error_per_sample0 + td_error_per_sample1) / 2
        else:
            q_data = v_1step_td_data(q_value, target_q_value, reward, done, data['weight'])
            loss_dict['critic_loss'], td_error_per_sample = v_1step_td_error(q_data, self._gamma)

        # 4. update q network
        self._optimizer_q.zero_grad()
        loss_dict['critic_loss'].backward()
        if self._twin_critic:
            loss_dict['twin_critic_loss'].backward()
        self._optimizer_q.step()

        # 5. evaluate to get action distribution
        (mu, sigma) = self._learn_model.forward(data['obs'], mode='compute_actor')['logit']
        dist = Independent(Normal(mu, sigma), 1)
        pred = dist.rsample()
        action = torch.tanh(pred)
        y = 1 - action.pow(2) + 1e-6
        # keep dimension for loss computation (usually for action space is 1 env. e.g. pendulum)
        log_prob = dist.log_prob(pred).unsqueeze(-1)
        log_prob = log_prob - torch.log(y).sum(-1, keepdim=True)

        # use the Kl divergence between actor and prior network to replace the original entropy term 
        if self._cfg.prior_kl and not (self._cfg.iter_turnoff_prior != 0 and train_iter > self._cfg.iter_turnoff_prior):
            no_prior_log_prob = log_prob

            (prior_mu, prior_sigma) = self.prior_network.forward(data['obs'], 'compute_actor')['logit']
            prior_dist = Independent(Normal(prior_mu, prior_sigma), 1)
            prior_log_prob = prior_dist.log_prob(pred).unsqueeze(-1)
            prior_log_prob = prior_log_prob - torch.log(y).sum(-1, keepdim=True)

            log_prob = no_prior_log_prob - prior_log_prob

        # selective Kl divergence, determined by which q value is higher between the prior network and the RL actor 
        if self._cfg.prior_filter_kl and not (self._cfg.iter_turnoff_prior != 0 and train_iter > self._cfg.iter_turnoff_prior):
            no_prior_log_prob = log_prob

            (prior_mu, prior_sigma) = self.prior_network.forward(data['obs'], 'compute_actor')['logit']
            prior_dist = Independent(Normal(prior_mu, prior_sigma), 1)
            prior_log_prob = prior_dist.log_prob(pred).unsqueeze(-1)
            prior_log_prob = prior_log_prob - torch.log(y).sum(-1, keepdim=True)
            prior_log_prob = no_prior_log_prob - prior_log_prob

            # the Q value of the action from the prior network
            (prior_mu, prior_sigma) = self.prior_network.forward(data['obs'], 'compute_actor')['logit']
            prior_dist = Independent(Normal(prior_mu, prior_sigma), 1)
            prior_pred = prior_dist.rsample()
            prior_action = torch.tanh(prior_pred)
            temp_prior_data = {'obs': obs, 'action': prior_action}
            prior_q_value = self._learn_model.forward(temp_prior_data, mode='compute_critic')['q_value']
            prior_q_value_min = torch.unsqueeze(torch.min(torch.vstack((prior_q_value[0], prior_q_value[1])), 0)[0], 1)

            # the Q value of the action from the RL actor network
            (mu, sigma) = self._learn_model.forward(data['obs'], mode='compute_actor')['logit']
            dist = Independent(Normal(mu, sigma), 1)
            pred = dist.rsample()
            action = torch.tanh(pred)
            temp_actor_data = {'obs': obs, 'action': action}
            actor_q_value = self._learn_model.forward(temp_actor_data, mode='compute_critic')['q_value']
            actor_q_value_min = torch.unsqueeze(torch.min(torch.vstack((actor_q_value[0], actor_q_value[1])), 0)[0], 1)

            # selective KL
            log_prob = torch.where(prior_q_value_min > actor_q_value_min, prior_log_prob, no_prior_log_prob)

            # some counters
            proir_num += torch.where(prior_q_value_min > actor_q_value_min)[0].shape[0]
            all_num += prior_q_value_min.shape[0]
            prior_ratior = proir_num / all_num
            loss_dict['prior_ratior'] = prior_ratior

        eval_data = {'obs': obs, 'action': action}
        new_q_value = self._learn_model.forward(eval_data, mode='compute_critic')['q_value']
        if self._twin_critic: # True
            new_q_value = torch.min(new_q_value[0], new_q_value[1])

        # 6. (optional) compute value loss and update value network
        if self._value_network: # False
            # new_q_value: (bs, ), log_prob: (bs, act_shape) -> target_v_value: (bs, )
            target_v_value = (new_q_value.unsqueeze(-1) - self._alpha * log_prob).mean(dim=-1)
            loss_dict['value_loss'] = F.mse_loss(v_value, target_v_value.detach())

            # update value network
            self._optimizer_value.zero_grad()
            loss_dict['value_loss'].backward()
            self._optimizer_value.step()

        # 7. compute policy loss
        policy_loss = (self._alpha * log_prob - new_q_value.unsqueeze(-1)).mean()

        loss_dict['policy_loss'] = policy_loss

        # 8. update policy network
        self._optimizer_policy.zero_grad()
        loss_dict['policy_loss'].backward()
        if self._cfg.pretraining and train_iter < self._cfg.pretraining_iter:
            pass
        else:
            self._optimizer_policy.step()

        # 9. compute alpha loss
        if self._cfg.pretraining and train_iter < self._cfg.pretraining_iter:
            pass
        else:
            if self._auto_alpha:    # True
                if self._log_space: # True
                    log_prob = log_prob + self._target_entropy
                    loss_dict['alpha_loss'] = -(self._log_alpha * log_prob.detach()).mean()

                    self._alpha_optim.zero_grad()
                    loss_dict['alpha_loss'].backward()
                    self._alpha_optim.step()
                    self._alpha = self._log_alpha.detach().exp()
                else:
                    log_prob = log_prob + self._target_entropy
                    loss_dict['alpha_loss'] = -(self._alpha * log_prob.detach()).mean()

                    self._alpha_optim.zero_grad()
                    loss_dict['alpha_loss'].backward()
                    self._alpha_optim.step()
                    self._alpha = max(0, self._alpha)

        loss_dict['total_loss'] = sum(loss_dict.values())

        # =============
        # after update
        # =============
        self._forward_learn_cnt += 1
        # target update
        self._target_model.update(self._learn_model.state_dict())
        return {
            'cur_lr_q': self._optimizer_q.defaults['lr'],
            'cur_lr_p': self._optimizer_policy.defaults['lr'],
            'priority': td_error_per_sample.abs().tolist(),
            'td_error': td_error_per_sample.detach().mean().item(),
            'alpha': self._alpha.item(),
            'target_q_value': target_q_value.detach().mean().item(),
            **loss_dict
        }