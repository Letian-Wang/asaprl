from easydict import EasyDict

main_config = dict(
    exp_name='asaprl',
    env=dict(
        manager=dict(
            episode_num=float('inf'),
            max_retry=5,
            retry_type='renew',
            auto_reset=True,
            step_timeout=None,
            reset_timeout=None,
            retry_waiting_time=0.1,
            cfg_type='BaseEnvManagerDict',
            shared_memory=False,
            context='spawn',
        ),
        metadrive={'use_render': False, 'show_seq_traj': False, 'use_jerk_penalty': True, 'traffic_density': 0.3, 'seq_traj_len': 10, 'use_lateral': True, 'DECODER_LOAD_DIR': 'pretrain_ckpt_files/seq_len_10_decoder_ckpt', 'vae_latent_dim': 2, 'traj_mode': 'decoder', 'lat_range': 30, 'camera_height': 70, 'use_chase_camera_follow_lane': True, 'reward_w_pass_car': 0.1, 'reward_w_on_lane': 0, 'reward_w_out_of_road': -5, 'reward_w_crash': -5, 'reward_w_destination': 0, 'reward_w_out_of_time': 0, 'reward_w_progress': 1, 'reward_w_speed': 0},
        n_evaluator_episode=20,
        stop_value=99999,
        collector_env_num=1,
        evaluator_env_num=1,
    ),
    policy=dict(
        model=dict(
            twin_critic=True,
            action_space='reparameterization',
            obs_shape=[5, 200, 200],
            action_shape=2,
            encoder_hidden_size_list=[128, 128, 64],
        ),
        learn=dict(
            learner=dict(
                train_iterations=1000000000,
                dataloader=dict(
                    num_workers=0,
                ),
                hook=dict(
                    load_ckpt_before_run='',
                    log_show_after_iter=100,
                    save_ckpt_after_iter=10000,
                    save_ckpt_after_run=True,
                ),
                cfg_type='BaseLearnerDict',
            ),
            multi_gpu=False,
            update_per_collect=100,
            batch_size=64,
            learning_rate_q=0.0003,
            learning_rate_policy=0.0003,
            learning_rate_value=0.0003,
            learning_rate_alpha=0.0003,
            target_theta=0.005,
            discount_factor=0.99,
            alpha=0.2,
            auto_alpha=True,
            log_space=True,
            ignore_done=False,
            init_w=0.003,
            learning_rate=0.0003,
        ),
        collect=dict(
            collector=dict(
                deepcopy_obs=False,
                transform_obs=False,
                collect_print_freq=100,
                cfg_type='MetadriveCollectorDict',
            ),
            collector_logit=False,
            n_sample=5000,
            unroll_len=1,
        ),
        eval=dict(
            evaluator=dict(
                eval_freq=1000,
                cfg_type='InteractionSerialEvaluatorDict',
                stop_value=99999,
                n_episode=20,
            ),
        ),
        other=dict(
            replay_buffer=dict(
                type='naive',
                replay_buffer_size=100000,
                deepcopy=False,
                enable_track_used_data=False,
                periodic_thruput_seconds=60,
                cfg_type='NaiveReplayBufferDict',
            ),
        ),
        cuda=True,
        on_policy=False,
        priority=False,
        priority_IS_weight=False,
        random_collect_size=10000,
        multi_agent=False,
        cfg_type='SACPolicyDict',
    ),
)
main_config = EasyDict(main_config)
main_config = main_config
create_config = dict(
    env=dict(
    ),
    env_manager=dict(
        cfg_type='BaseEnvManagerDict',
    ),
    policy=dict(type='sac'),
)
create_config = EasyDict(create_config)
create_config = create_config
