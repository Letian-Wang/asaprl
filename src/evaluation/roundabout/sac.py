import metadrive, argparse, os, torch
from easydict import EasyDict
from functools import partial
from tensorboardX import SummaryWriter

from metadrive import TopDownMetaDrive
from ding.envs import BaseEnvManager, BaseEnvManager
from ding.config import compile_config
from ding.policy import SACPolicy
from ding.worker import SampleSerialCollector, InteractionSerialEvaluator, BaseLearner, NaiveReplayBuffer
from asaprl.envs import DriveEnvWrapper
from asaprl.policy.conv_qac import ConvQAC
from asaprl.envs.control_env import MetaDriveControlEnv
from asaprl.utils.rl_utils.evaluator_utils import MetadriveEvaluator
from asaprl.utils.env_utils.reward_utils import reward_weight

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default='sac_roundabout')                   # the directory of the trained model
parser.add_argument('--ckpt_file', type=str, default='ckpt/iteration_80000.pth.tar')    # the directory of the trained ckpt, under the path of trained model
parser.add_argument('--dense_reward', type=bool, default=False)             # whether use the dense reward setting or not
parser.add_argument('--reward_version', type=int, default=27)               # different reward weight
args = parser.parse_args()
metadrive_basic_config = dict(
    exp_name=args.exp_name,
    env=dict(
        metadrive=dict(
            use_render=True,
            seq_traj_len = 1,
            use_jerk_penalty = True,
            use_lateral_penalty = False,
            traffic_density = 0.3,
            half_jerk = False,
            map='OSOS', 
            use_lateral = True, 
            dense_reward = args.dense_reward,
            show_interface = False,
            camera_height = 70,
            use_chase_camera_follow_lane = True
            ),
        manager=dict(
            shared_memory=False,
            max_retry=2,
            context='spawn',
        ),
        n_evaluator_episode=10,
        stop_value=99999,
        collector_env_num=1,
        evaluator_env_num=1,
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
            n_sample=5000,
        ),
        eval=dict(
            evaluator=dict(
                eval_freq=1000,
                )
            ),
        other=dict(
            replay_buffer=dict(
                replay_buffer_size=100000,
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
        BaseEnvManager,
        SACPolicy,
        BaseLearner,
        SampleSerialCollector,
        InteractionSerialEvaluator,
        NaiveReplayBuffer,
    )
    evaluator_env = BaseEnvManager(
        env_fn=[partial(wrapped_env, cfg.env.metadrive) for _ in range(cfg.env.evaluator_env_num)],
        cfg=cfg.env.manager,
    )

    model = ConvQAC(**cfg.policy.model)
    policy = SACPolicy(cfg.policy, model=model)
    load_model_path = os.path.join(args.exp_name, args.ckpt_file)
    if os.path.exists(load_model_path):
        print('loading path: ', load_model_path)
        trained_model = torch.load(load_model_path, map_location=torch.device('cpu'))
        policy._load_state_dict_learn(trained_model) 

    tb_logger = SummaryWriter('./log/{}/'.format(cfg.exp_name))
    evaluator = MetadriveEvaluator(cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name)

    stop, rate = evaluator.eval()
    evaluator.close()


if __name__ == '__main__':
    main(main_config)
