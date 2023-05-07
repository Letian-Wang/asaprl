
def reward_weight(reward_version, config_dict):
    if reward_version == 27:
        config_dict['env']['metadrive']['reward_w_pass_car'] = 0.1
        config_dict['env']['metadrive']['reward_w_on_lane'] = 0
        config_dict['env']['metadrive']['reward_w_out_of_road'] = -5
        config_dict['env']['metadrive']['reward_w_crash'] = -5
        config_dict['env']['metadrive']['reward_w_destination'] = 0
        config_dict['env']['metadrive']['reward_w_out_of_time'] = 0
        config_dict['env']['metadrive']['reward_w_progress'] = 1
        config_dict['env']['metadrive']['reward_w_speed'] = 0
    elif reward_version == 215:
        config_dict['env']['metadrive']['reward_w_pass_car'] = 0.1
        config_dict['env']['metadrive']['reward_w_on_lane'] = 0
        config_dict['env']['metadrive']['reward_w_out_of_road'] = -5
        config_dict['env']['metadrive']['reward_w_crash'] = -5
        config_dict['env']['metadrive']['reward_w_destination'] = 1
        config_dict['env']['metadrive']['reward_w_out_of_time'] = 0
        config_dict['env']['metadrive']['reward_w_progress'] = 1
        config_dict['env']['metadrive']['reward_w_speed'] = 0
    return config_dict

