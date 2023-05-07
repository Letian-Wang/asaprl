import metadrive
from easydict import EasyDict
from functools import partial
from tensorboardX import SummaryWriter
import argparse, pdb
import torch
from ding.envs import SyncSubprocessEnvManager
from ding.config import compile_config
from asaprl.policy.prior_sac import PriorSAC
from ding.worker import InteractionSerialEvaluator, BaseLearner, NaiveReplayBuffer
from asaprl.envs import DriveEnvWrapper
from asaprl.policy.conv_qac import ConvQAC
from asaprl.envs.skill_env import MetaDriveSkillEnv
from asaprl.utils.rl_utils.evaluator_utils import MetadriveEvaluator, clean_old_model
from asaprl.utils.rl_utils.collector_utils import MetadriveCollector
from asaprl.utils.rl_utils.learner_utils import SkillLearner
from asaprl.utils.env_utils.reward_utils import reward_weight

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default='asaprl_roundabout')
# parameters related to the environment
parser.add_argument('--dense_reward', type=bool, default=False)             # whether use the dense reward setting or not
parser.add_argument('--reward_version', type=int, default=0)                    # different reward weight
parser.add_argument('--reward_average_length', type=bool, default=False)        # whether devide the reward with the skill length, for one skill execution
parser.add_argument('--traffic_density', type=float, default=0.3)               # the traffic density on the scene
# parameters related to training
parser.add_argument('--alpha', type=float, default=0.2)                         # the initial alpha value in the SAC training
parser.add_argument('--save_ckpt_after_iter', type=int, default=10000)          # how frequent the ckpt is saved
# parameters related to expert prior
parser.add_argument('--ACTOR_PRIOR_LOAD_DIR', type=str, default=None)                 # the directory of the pretrained actor
parser.add_argument('--pretraining', type=bool, default=False)                  # whether use the pretrained actor to initialize the actor
parser.add_argument('--pretraining_iter', type=int, default=0)                  # how many iterations that the actor is freezed at the early stage of training
parser.add_argument('--reward_augment', type=bool, default=False)               # whether use the expert prior as an additional reward term
parser.add_argument('--prior_kl', type=bool, default=False)                     # whether use the expert prior as a KL divergence term to replace the entropy term in SAC
parser.add_argument('--prior_filter_kl', type=bool, default=False)              # wheter use the expert prior as a KL divergence term to replace the entropy term in SAC, when the Q value of the expert prior's action is higher than the Q value of the actor's action
parser.add_argument('--prior_prob', type=bool, default=False)                   # whether use the expert prior as a probability term ro replace the entropy term in SAC
parser.add_argument('--iter_turnoff_prior', type=int, default=0)                # turn off the KL divergence term after this iteration parameter
parser.add_argument('--iter_gradual_prior', type=int, default=0)                # gradually turn off the KL divergence term, which is turned off completely after this iteration parameter
parser.add_argument('--CRITIC_PRIOR_LOAD_DIR', type=str, default='pretrain_ckpt_files/prior_network_RL_expert/roundabout_pretrainwarm2000_iteration_3000.pth.tar') # the directory of the pretrained critic model, which is a whole ckpt including both critic, actor, and target network
parser.add_argument('--critic_pretraining', type=bool, default=False)           # whether use the pretrained actor to initialize the actor
# parameters related to the skill
parser.add_argument('--lat_range', type=float, default=30)                      # scaling parameters, used to scale the value output by actor to the lateral parameter for motion skill generation
parser.add_argument('--traj_mode', type=str, default='decoder')                 # different trajectory generation mode, 'decoder' for SPiRL and TaEcRL, 'planningfixed' for ASAPRL, 'planningvariable' for variable skill length
parser.add_argument('--action_shape', type=int, default=2)                      # action shape of the policy
parser.add_argument('--SEQ_TRAJ_LEN', type=int, default=10)                     # skill length of the policy
parser.add_argument('--DECODER_LOAD_DIR', type=str, default='pretrain_ckpt_files/seq_len_10_decoder_ckpt')   # the directory of the trained trajecotyr decoder, for SPiRL and TaEcRL
args = parser.parse_args()

metadrive_basic_config = dict(
    exp_name = args.exp_name,
    env=dict(
        metadrive=dict(
            use_render=False,
            traffic_density = args.traffic_density,
            seq_traj_len = args.SEQ_TRAJ_LEN,
            use_jerk_penalty = True,                            # dense reward term
            use_lateral = True,                                 # dense reward term
            dense_reward = args.dense_reward,                   # dense reward term
            DECODER_LOAD_DIR = args.DECODER_LOAD_DIR,
            vae_latent_dim = args.action_shape,
            traj_mode = args.traj_mode,
            lat_range = args.lat_range,
            map='OSOS', 
            reward_average_length = args.reward_average_length
            ),
        manager=dict(
            shared_memory=False,
            max_retry=5,
            retry_type='renew',
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
            action_shape=args.action_shape,
            encoder_hidden_size_list=[128, 128, 64],
        ),
        ACTOR_PRIOR_LOAD_DIR = args.ACTOR_PRIOR_LOAD_DIR,
        reward_augment = args.reward_augment,
        prior_kl = args.prior_kl,
        prior_filter_kl = args.prior_filter_kl,
        prior_prob = args.prior_prob,
        iter_turnoff_prior = args.iter_turnoff_prior,
        iter_gradual_prior = args.iter_gradual_prior,
        pretraining = args.pretraining,
        pretraining_iter = args.pretraining_iter,
        learn=dict(
            update_per_collect=100,
            batch_size=128,
            learning_rate=3e-4,
            alpha = args.alpha,
            learner=dict(
                hook=dict(
                    save_ckpt_after_iter = args.save_ckpt_after_iter,
                    ),
                ),
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
    return DriveEnvWrapper(MetaDriveSkillEnv(config=env_cfg), wrapper_cfg)

def main(cfg):
    cfg = compile_config(
        cfg,
        SyncSubprocessEnvManager,
        PriorSAC,
        SkillLearner,
        MetadriveCollector,
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
    policy = PriorSAC(cfg.policy, model=model)

    tb_logger = SummaryWriter('./log/{}/'.format(cfg.exp_name))
    learner = SkillLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    collector = MetadriveCollector(cfg.policy.collect.collector, collector_env, policy.collect_mode, tb_logger, exp_name=cfg.exp_name)
    evaluator = MetadriveEvaluator(cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name)
    replay_buffer = NaiveReplayBuffer(cfg.policy.other.replay_buffer, tb_logger, exp_name=cfg.exp_name)

    learner.call_hook('before_run')

    if args.critic_pretraining:
        if torch.cuda.is_available(): pretrained_model = torch.load(args.CRITIC_PRIOR_LOAD_DIR)
        else: pretrained_model = torch.load(args.CRITIC_PRIOR_LOAD_DIR, map_location=torch.device('cpu'))
        policy._model.load_state_dict(pretrained_model['model'])
        policy._target_model.load_state_dict(pretrained_model['target_model'])
    if cfg.policy.pretraining:
        if torch.cuda.is_available(): pretrained_actor = torch.load(cfg.policy.ACTOR_PRIOR_LOAD_DIR)
        else: pretrained_actor = torch.load(cfg.policy.ACTOR_PRIOR_LOAD_DIR, map_location=torch.device('cpu'))
        policy._model.actor.load_state_dict(pretrained_actor) 

    while True:
        if evaluator.should_eval(learner.train_iter):
            stop, rate = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
            clean_old_model(cfg.exp_name)
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