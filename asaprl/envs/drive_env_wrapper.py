import gym
import copy
import numpy as np
from typing import Any, Dict, Optional
from easydict import EasyDict
from itertools import product

from .base_drive_env import BaseDriveEnv
from ding.utils.default_helper import deep_merge_dicts
from ding.envs.env.base_env import BaseEnvTimestep
from ding.envs.common.env_element import EnvElementInfo
from ding.torch_utils.data_helper import to_ndarray


class DriveEnvWrapper(gym.Wrapper):
    """
    Environment wrapper to make ``gym.Env`` align with DI-engine definitions, so as to use utilities in DI-engine.
    It changes ``step``, ``reset`` and ``info`` method of ``gym.Env``, while others are straightly delivered.

    :Arguments:
        - env (BaseDriveEnv): The environment to be wrapped.
        - cfg (Dict): Config dict.

    :Interfaces: reset, step, info, render, seed, close
    """

    config = dict()

    def __init__(self, env: BaseDriveEnv, cfg: Dict = None, **kwargs) -> None:
        if cfg is None:
            self._cfg = self.__class__.default_config()
        elif 'cfg_type' not in cfg:
            self._cfg = self.__class__.default_config()
            self._cfg = deep_merge_dicts(self._cfg, cfg)
        else:
            self._cfg = cfg
        self.env = env
        if not hasattr(self.env, 'reward_space'):
            self.reward_space = gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(1, ))

    def reset(self, *args, **kwargs) -> Any:
        """
        Wrapper of ``reset`` method in env. The observations are converted to ``np.ndarray`` and final reward
        are recorded.

        :Returns:
            Any: Observations from environment
        """
        # import pdb; pdb.set_trace()
        obs = self.env.reset(*args, **kwargs)
        obs = to_ndarray(obs, dtype=np.float32)
        if isinstance(obs, np.ndarray) and len(obs.shape) == 3:
            obs = obs.transpose((2, 0, 1))
        self._final_eval_reward = 0.0
        return obs

    def step(self, action: Any = None) -> BaseEnvTimestep:
        """
        Wrapper of ``step`` method in env. This aims to convert the returns of ``gym.Env`` step method into
        that of ``ding.envs.BaseEnv``, from ``(obs, reward, done, info)`` tuple to a ``BaseEnvTimestep``
        namedtuple defined in DI-engine. It will also convert actions, observations and reward into
        ``np.ndarray``, and check legality if action contains control signal.

        :Arguments:
            - action (Any, optional): Actions sent to env. Defaults to None.

        :Returns:
            BaseEnvTimestep: DI-engine format of env step returns.
        """
        action = to_ndarray(action)

        obs, rew, done, info = self.env.step(action)
        self._final_eval_reward += rew
        obs = to_ndarray(obs, dtype=np.float32)
        if isinstance(obs, np.ndarray) and len(obs.shape) == 3:
            obs = obs.transpose((2, 0, 1))
        rew = to_ndarray([rew], dtype=np.float32)
        if done:
            info['final_eval_reward'] = self._final_eval_reward
        return BaseEnvTimestep(obs, rew, done, info)

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def enable_save_replay(self, replay_path: Optional[str] = None) -> None:
        if replay_path is None:
            replay_path = './video'
        self._replay_path = replay_path
        self.env = gym.wrappers.Monitor(self.env, self._replay_path, video_callable=lambda episode_id: True, force=True)

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(cls.config)
        cfg.cfg_type = cls.__name__ + 'Config'
        return copy.deepcopy(cfg)

    def __repr__(self) -> str:
        return repr(self.env)

    def render(self):
        self.env.render()

