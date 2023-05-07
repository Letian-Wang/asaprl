from torch.utils.data import Dataset
import pdb
import os
import pickle
import random
import torch
import numpy as np

# load raw demonstration
class DemoDataset(Dataset):
    def __init__(self, scenario, data_mode = 'train', skill_length = 10):
        self.data_mode = data_mode 
        self.scenario = scenario
        self.split_ratio = 0.9
        self.skill_length = skill_length
        self.data_path = './demonstration_RL_expert/{}/'.format(self.scenario)
        self.extraced_data = self.read_all_data()
        self.len = len(self.extraced_data)


    def read_all_data(self):
        all_data = []

        all_file_lst = os.listdir(self.data_path)
        for one_file in all_file_lst:
            one_file_full_path = self.data_path + one_file
            with open(one_file_full_path, 'rb') as handle:
                one_file_data = pickle.load(handle)
            for latent_var_idx, one_latent_var in enumerate(one_file_data['latent_var']):
                one_obs = one_file_data['obs'][latent_var_idx * self.skill_length]
                one_obs = np.transpose(one_obs, (2, 0, 1))
                one_logit = one_file_data['logit'][latent_var_idx] if 'logit' in one_file_data.keys() else (1,1)
                all_data.append({'obs': one_obs, 'latent_var': one_latent_var, 'logit': one_logit})

        ''' split training and validation data '''
        all_data_index = [i for i in range(len(all_data))]
        training_data_index = random.sample(all_data_index, int(len(all_data_index)* self.split_ratio))
        if self.data_mode == 'train':
            train_data = []
            for one_index in training_data_index:
                train_data.append(all_data[one_index])
            return train_data
        elif self.data_mode == 'val':
            val_data = []
            for one_index in all_data_index:
                if one_index not in training_data_index:
                    val_data.append(all_data[one_index])
            return val_data

    def __len__(self):
        return self.len 
    
    def __getitem__(self, idx):
        return self.extraced_data[idx]['obs'], self.extraced_data[idx]['latent_var'], self.extraced_data[idx]['logit']
def create_raw_data_loader(scenario, batch_size):
    train_dataset = DemoDataset(scenario=scenario, data_mode = 'train')
    val_dataset = DemoDataset(scenario=scenario, data_mode = 'val')

    train_loader = torch.utils.data.DataLoader(dataset= train_dataset, batch_size = batch_size, drop_last=True, shuffle = True, pin_memory=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(dataset= val_dataset, batch_size = batch_size, drop_last=True, shuffle = True, pin_memory=True, num_workers=0)

    print("train dataset size: ", len(train_dataset), len(train_dataset)/batch_size)
    print("val dataset size: ", len(val_dataset), len(val_dataset)/batch_size)
    print("train loader size: ", len(train_loader))
    print("train loader size: ", len(val_loader))

    return train_loader, val_loader


# load annotated demonstration
class AnnotatedDataset(Dataset):
    def __init__(self, scenario, data_mode = 'train', skill_length = 10):
        self.data_mode = data_mode 
        self.scenario = scenario
        self.split_ratio = 0.9
        self.skill_length = skill_length
        self.data_path = './demonstration_RL_expert/{}_annotated/'.format(self.scenario)
        self.extraced_data = self.read_all_data()
        self.len = len(self.extraced_data)

    def read_all_data(self):
        '''
        We use the trained RL agent to collect data, specifically, the RL agent we used is the 'No Prior' version of our method.
        The collect data has the ground-truth skill parameters, which enable us to examine the accuracy of the recovered parameters. 
        When we use other agenets, such ground-truth skill parameters will be not available
        '''
        all_data = []
        ''' record data to examine recovery accuracy '''
        # latent variable output from the policy, which is range of [0, 1]
        errors_latent_var0 = []  # 0.15123                             # 0.04692
        errors_latent_var1 = []  # 0.13120                             # 0.04014
        errors_latent_var2 = []  # 0.03711                             # 8.44 e-05
        # planning parameters, which is transformed from latent variable
        errors_lat = []  # 0.75617                             # 0.23463
        errors_yaw = []  # 3.93600                             # 1.20443
        errors_v = []    # 0.18557                             # 0.00042
        traj_errors = []


        all_file_lst = os.listdir(self.data_path)
        for one_file in all_file_lst:
            one_file_full_path = self.data_path + one_file
            with open(one_file_full_path, 'rb') as handle:
                one_file_data = pickle.load(handle)
            for latent_var_idx, one_latent_var in enumerate(one_file_data['latent_var']):
                one_obs = one_file_data['obs'][latent_var_idx * self.skill_length]
                one_obs = np.transpose(one_obs, (2, 0, 1))
                one_recovered_latent_var = one_file_data['recovered_latent_var'][latent_var_idx]
                one_logit = one_file_data['logit'][latent_var_idx] if 'logit' in one_file_data.keys() else (1,1)
                all_data.append({'obs': one_obs, 'latent_var': one_latent_var, 'logit': one_logit, 'recovered_latent_var': one_recovered_latent_var})

                ''' record data to examine recovery accuracy '''
                # errors_latent_var0.append(abs(one_recovered_latent_var - one_latent_var)[0])
                # errors_latent_var1.append(abs(one_recovered_latent_var - one_latent_var)[1])
                # errors_latent_var2.append(abs(one_recovered_latent_var - one_latent_var)[2])
                # errors_lat.append(abs(one_recovered_latent_var - one_latent_var)[0] * 5)
                # errors_yaw.append(abs(one_recovered_latent_var - one_latent_var)[1] * 30)
                # errors_v.append(abs(one_recovered_latent_var - one_latent_var)[2] * 5)
                # pdb.set_trace()

        ''' split training and validation data '''
        all_data_index = [i for i in range(len(all_data))]
        training_data_index = random.sample(all_data_index, int(len(all_data_index)* self.split_ratio))
        if self.data_mode == 'train':
            train_data = []
            for one_index in training_data_index:
                train_data.append(all_data[one_index])
            return train_data
        elif self.data_mode == 'val':
            val_data = []
            for one_index in all_data_index:
                if one_index not in training_data_index:
                    val_data.append(all_data[one_index])
            return val_data

    def __len__(self):
        return self.len 
    
    def __getitem__(self, idx):
        return self.extraced_data[idx]['obs'], self.extraced_data[idx]['latent_var'], self.extraced_data[idx]['logit'], self.extraced_data[idx]['recovered_latent_var']
def create_annotated_data_loader(scenario, batch_size):
    train_dataset = AnnotatedDataset(scenario=scenario, data_mode = 'train')
    val_dataset = AnnotatedDataset(scenario=scenario, data_mode = 'val')

    train_loader = torch.utils.data.DataLoader(dataset= train_dataset, batch_size = batch_size, drop_last=True, shuffle = True, pin_memory=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(dataset= val_dataset, batch_size = batch_size, drop_last=True, shuffle = True, pin_memory=True, num_workers=0)

    print("train dataset size: ", len(train_dataset), len(train_dataset)/batch_size)
    print("val dataset size: ", len(val_dataset), len(val_dataset)/batch_size)
    print("train loader size: ", len(train_loader))
    print("train loader size: ", len(val_loader))

    return train_loader, val_loader
