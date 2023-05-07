import metadrive, argparse
from easydict import EasyDict
from functools import partial
from tensorboardX import SummaryWriter

from metadrive import TopDownMetaDrive
from ding.envs import BaseEnvManager, SyncSubprocessEnvManager
from ding.config import compile_config
from ding.policy import SACPolicy
from ding.worker import SampleSerialCollector, InteractionSerialEvaluator, BaseLearner, NaiveReplayBuffer
from asaprl.envs import DriveEnvWrapper
from asaprl.policy.conv_qac import ConvQAC
from asaprl.envs.control_env import MetaDriveControlEnv
from asaprl.utils.rl_utils.evaluator_utils import MetadriveEvaluator
from asaprl.utils.env_utils.reward_utils import reward_weight

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default='taecrl')
parser.add_argument('--dense_reward', type=bool, default=False)             # whether use the dense reward setting or not
parser.add_argument('--reward_version', type=int, default=27)               # different reward weight
args = parser.parse_args()
metadrive_basic_config = dict(
    exp_name=args.exp_name,
    env=dict(
        metadrive=dict(
            use_render=False,
            seq_traj_len = 10,
            traffic_density = 0.3,
            const_control = True,
            map='OSOS', 
            use_jerk_penalty = True,                    # dense reward term
            use_lateral_penalty = False,                # dense reward term
            half_jerk = False,                          # dense reward term
            use_lateral = True,                         # dense reward term
            dense_reward = args.dense_reward
            ),
        manager=dict(
            shared_memory=False,
            max_retry=2,
            context='spawn',
        ),
        n_evaluator_episode=1, # 10
        stop_value=99999,
        collector_env_num=1, #12
        evaluator_env_num=1, # 2
    ),
    policy=dict(
        cuda=True,
        model=dict(
            obs_shape=[5, 200, 200],
            action_shape=2,
            encoder_hidden_size_list=[128, 128, 64],
        ),
        learn=dict(
            update_per_collect=100,
            batch_size=64,
            learning_rate=3e-4,
        ),
        collect=dict(
            n_sample=200, # 5000
        ),
        eval=dict(
            evaluator=dict(
                eval_freq=1000,
                )
            ),
        other=dict(
            replay_buffer=dict(
                replay_buffer_size=1000, # 100000
            ),
        ), 
    )
)
metadrive_basic_config = reward_weight(args.reward_version, metadrive_basic_config)
main_config = EasyDict(metadrive_basic_config)

def wrapped_env(env_cfg, wrapper_cfg=None):
    return DriveEnvWrapper(MetaDriveControlEnv(config=env_cfg), wrapper_cfg)


def main(cfg):
    cfg = compile_config(
        cfg,
        SyncSubprocessEnvManager,
        SACPolicy,
        BaseLearner,
        SampleSerialCollector,
        InteractionSerialEvaluator,
        NaiveReplayBuffer,
    )

    collector_env_num, evaluator_env_num = cfg.env.collector_env_num, cfg.env.evaluator_env_num
    collector_env = SyncSubprocessEnvManager(
        env_fn=[partial(wrapped_env, cfg.env.metadrive) for _ in range(collector_env_num)],
        cfg=cfg.env.manager,
    )
    evaluator_env = SyncSubprocessEnvManager(
        env_fn=[partial(wrapped_env, cfg.env.metadrive) for _ in range(evaluator_env_num)],
        cfg=cfg.env.manager,
    )

    model = ConvQAC(**cfg.policy.model)
    policy = SACPolicy(cfg.policy, model=model)

    tb_logger = SummaryWriter('./log/{}/'.format(cfg.exp_name))
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    collector = SampleSerialCollector(cfg.policy.collect.collector, collector_env, policy.collect_mode, tb_logger, exp_name=cfg.exp_name)
    evaluator = MetadriveEvaluator(cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name)
    replay_buffer = NaiveReplayBuffer(cfg.policy.other.replay_buffer, tb_logger, exp_name=cfg.exp_name)

    learner.call_hook('before_run')

    while True:
        if evaluator.should_eval(learner.train_iter):
            stop, rate = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep, dense_reward = args.dense_reward)
            if stop:
                break
        # Sampling data from environments
        new_data = collector.collect(cfg.policy.collect.n_sample, train_iter=learner.train_iter)
        replay_buffer.push(new_data, cur_collector_envstep=collector.envstep)
        for i in range(cfg.policy.learn.update_per_collect):
            train_data = replay_buffer.sample(learner.policy.get_attribute('batch_size'), learner.train_iter)
            if train_data is None:
                break
            learner.train(train_data, collector.envstep)
    learner.call_hook('after_run')

    collector.close()
    evaluator.close()
    learner.close()


if __name__ == '__main__':
    main(main_config)
