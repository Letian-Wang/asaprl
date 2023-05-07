import torch
import os

class hyper_parameter(object):
    def __init__(self, args):
        # dataset parameter
        self.exp_name = args.exp_name
        self.KL_weight = args.KL_weight

        self.annotate_skill = args.annotate_skill
        
        self.learning_rate = args.learning_rate

        self.action_shape = args.action_shape
        self.scenario = args.scenario
        self.batch_size = args.batch_size
        self.n_epochs = args.n_epochs

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = "cpu"

        self.val_freq = 10

        self.save_freq = 10

    def mkdir_write_params(self):
        if not os.path.exists('./saved_model/'): os.mkdir('./saved_model/')
        if not os.path.exists('./saved_model/pretrain_actor/'): os.mkdir('./saved_model/pretrain_actor/')
        if not os.path.exists('./saved_model/pretrain_actor/log'): os.mkdir('./saved_model/pretrain_actor/log')
        if not os.path.exists('./saved_model/pretrain_actor/log/{}'.format(self.exp_name)): os.mkdir('./saved_model/pretrain_actor/log/{}'.format(self.exp_name))
        if not os.path.exists('./saved_model/pretrain_actor/{}'.format(self.exp_name)): os.mkdir('./saved_model/pretrain_actor/{}'.format(self.exp_name))
        if not os.path.exists("./saved_model/pretrain_actor/{}/ckpt".format(self.exp_name)): os.mkdir("saved_model/pretrain_actor/{}/ckpt".format(self.exp_name))
        with open("./saved_model/pretrain_actor/{}/config.txt".format(self.exp_name), 'w') as filestream:
            for key in self.__dict__.keys():
                filestream.write("{}: {}\n".format(key, self.__dict__[key]))
    
    def mkdir_write_params_gt(self):
        if not os.path.exists('./saved_model/'): os.mkdir('./saved_model/')
        if not os.path.exists('./saved_model/pretrain_actor_gt_skill/'): os.mkdir('./saved_model/pretrain_actor_gt_skill/')
        if not os.path.exists('./saved_model/pretrain_actor_gt_skill/log'): os.mkdir('./saved_model/pretrain_actor_gt_skill/log')
        if not os.path.exists('./saved_model/pretrain_actor_gt_skill/log/{}'.format(self.exp_name)): os.mkdir('./saved_model/pretrain_actor_gt_skill/log/{}'.format(self.exp_name))
        if not os.path.exists('./saved_model/pretrain_actor_gt_skill/{}'.format(self.exp_name)): os.mkdir('./saved_model/pretrain_actor_gt_skill/{}'.format(self.exp_name))
        if not os.path.exists("./saved_model/pretrain_actor_gt_skill/{}/ckpt".format(self.exp_name)): os.mkdir("saved_model/pretrain_actor_gt_skill/{}/ckpt".format(self.exp_name))
        with open("./saved_model/pretrain_actor_gt_skill/{}/config.txt".format(self.exp_name), 'w') as filestream:
            for key in self.__dict__.keys():
                filestream.write("{}: {}\n".format(key, self.__dict__[key]))
        