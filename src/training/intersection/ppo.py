import metadrive, argparse
from easydict import EasyDict
from functools import partial
from tensorboardX import SummaryWriter

from metadrive import TopDownMetaDrive
from ding.envs import BaseEnvManager, SyncSubprocessEnvManager
from ding.config import compile_config
from ding.policy import PPOPolicy
from ding.worker import SampleSerialCollector, InteractionSerialEvaluator, BaseLearner
from asaprl.envs import DriveEnvWrapper
from asaprl.policy.conv_vac import ConvVAC
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
            seq_traj_len = 1,
            traffic_density = 0.3,
            map='XSXS', 
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
        action_space='continuous',
        model=dict(
            obs_shape=[5, 200, 200],
            action_shape=2,
            action_space='continuous',
            encoder_hidden_size_list=[128, 128, 64],
        ),
        learn=dict(
            epoch_per_collect=2,
            batch_size=64,
            learning_rate=3e-4,
            value_weight = 0.5,
            adv_norm=False,
            value_norm=True,
            grad_clip_value=10
        ),
        collect=dict(
            n_sample=200, # 1000
        ),
        eval=dict(
            evaluator=dict(
                eval_freq=1000,
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
        PPOPolicy,
        BaseLearner,
        SampleSerialCollector,
        InteractionSerialEvaluator,
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

    model = ConvVAC(**cfg.policy.model)
    policy = PPOPolicy(cfg.policy, model=model)

    tb_logger = SummaryWriter('./log/{}/'.format(cfg.exp_name))
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    collector = SampleSerialCollector(
        cfg.policy.collect.collector, collector_env, policy.collect_mode, tb_logger, exp_name=cfg.exp_name
    )
    evaluator = MetadriveEvaluator(
        cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name
    )

    learner.call_hook('before_run')

    while True:
        if evaluator.should_eval(learner.train_iter):
            print("evaluating")
            stop, rate = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep, dense_reward = args.dense_reward)
            if stop:
                break
        # Sampling data from environments
        print("collecting")
        new_data = collector.collect(cfg.policy.collect.n_sample, train_iter=learner.train_iter)
        print("learning")
        learner.train(new_data, collector.envstep)
    learner.call_hook('after_run')

    collector.close()
    evaluator.close()
    learner.close()


if __name__ == '__main__':
    main(main_config)
