from asaprl.policy.conv_qac import ConvQAC
from ding.policy import SACPolicy
import torch
import os
def get_prior_network(load_ckpt_path, action_shape):
    model_config = dict(
                obs_shape=[5, 200, 200],
                action_shape=action_shape,
                encoder_hidden_size_list=[128, 128, 64],
    )
    model_config2 = SACPolicy.config['model']
    model_config2.update(model_config)
    model = ConvQAC(**model_config2)

    if load_ckpt_path is not None and os.path.exists(load_ckpt_path):
        if torch.cuda.is_available():
            load_model = torch.load(load_ckpt_path)
        else:
            load_model = torch.load(load_ckpt_path, map_location=torch.device('cpu'))
        model.actor.load_state_dict(load_model)

    return model