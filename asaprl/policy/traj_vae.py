import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.distributions import Normal, Independent


''' 
The decoder that transforms latent skill variable into actual skill, as used in SPiRL and TaEcRL

'''
class VaeEncoder(nn.Module):
    def __init__(self,
        embedding_dim = 64,
        h_dim = 64,
        latent_dim = 100,
        seq_len = 30,
        use_relative_pos = True,
        dt = 0.03,
        ):
        super(VaeEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.h_dim = h_dim 
        self.num_layers = 1
        self.latent_dim = latent_dim
        self.seq_len = seq_len 
        self.use_relative_pos = use_relative_pos
        self.dt = dt
        self.device = torch.device('cuda:0')

        # input: x, y, theta, v,   output: embedding
        self.spatial_embedding = nn.Linear(4, self.embedding_dim)

        enc_mid_dims = [self.h_dim, self.h_dim, self.h_dim, self.latent_dim]
        mu_modules = []
        sigma_modules = []
        in_channels = self.h_dim 
        for m_dim in enc_mid_dims:
            mu_modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, m_dim),
                    nn.LeakyReLU())
            )
            sigma_modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, m_dim),
                    nn.LeakyReLU())
            )
            in_channels = m_dim  
        self.mean = nn.Sequential(*mu_modules) 
        self.log_var = nn.Sequential(*sigma_modules)
        self.encoder = nn.LSTM(self.embedding_dim, self.h_dim, self.num_layers)

    def init_hidden(self, batch_size):
        return (
            torch.zeros(self.num_layers, batch_size, self.h_dim).to(self.device),
            torch.zeros(self.num_layers, batch_size, self.h_dim).to(self.device)
        )
    def get_relative_position(self, abs_traj):
        # abs_traj shape: batch_size x seq_len x 4
        # rel traj shape: batch_size x seq_len -1 x 2
        rel_traj = abs_traj[:, 1:, :2] - abs_traj[:, :-1, :2]
        rel_traj = torch.cat([abs_traj[:, 0, :2].unsqueeze(1), rel_traj], dim = 1)
        rel_traj = torch.cat([rel_traj, abs_traj[:,:,2:]],dim=2)
        # rel_traj shape: batch_size x seq_len x 4
        return rel_traj
    
    def encode(self, input):
        # input meaning: a trajectory len 25 and contains x, y , theta, v
        # input shape: batch x seq_len x 4
        #data_traj shape: seq_len x batch x 4
        if self.use_relative_pos:
            input = self.get_relative_position(input)
        data_traj = input.permute(1, 0, 2).contiguous()
        traj_embedding = self.spatial_embedding(data_traj.view(-1, 4))
        traj_embedding = traj_embedding.view(self.seq_len, -1, self.embedding_dim)
        # Here we do not specify batch_size to self.batch_size because when testing maybe batch will vary
        batch_size = traj_embedding.shape[1]
        hidden_tuple = self.init_hidden(batch_size)
        output, encoder_h = self.encoder(traj_embedding, hidden_tuple)
        mu = self.mean(encoder_h[0])
        log_var = self.log_var(encoder_h[0])
        #mu, log_var = torch.tanh(mu), torch.tanh(log_var)
        return mu, log_var

    def forward(self, input):
        return self.encode(input)


class VaeDecoder(nn.Module):
    def __init__(self,
        embedding_dim = 64,
        h_dim = 64,
        latent_dim = 100,
        seq_len = 30,
        use_relative_pos = True,
        dt = 0.03,
        ):
        super(VaeDecoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.h_dim = h_dim 
        self.num_layers = 1
        self.latent_dim = latent_dim
        self.seq_len = seq_len 
        self.use_relative_pos = use_relative_pos
        self.dt = dt
        # input: x, y, theta, v,   output: embedding
        self.spatial_embedding = nn.Linear(4, self.embedding_dim)
        # input: h_dim, output: throttle, steer
        self.hidden2control = nn.Linear(self.h_dim, 2)
        self.decoder = nn.LSTM(self.embedding_dim, self.h_dim, self.num_layers)
        self.init_hidden_decoder = torch.nn.Linear(in_features = self.latent_dim, out_features = self.h_dim * self.num_layers)

    def plant_model_batch(self, prev_state_batch, pedal_batch, steering_batch, dt = 0.03):
        #import copy
        prev_state = prev_state_batch
        x_t = prev_state[:,0]
        y_t = prev_state[:,1]
        psi_t = prev_state[:,2]
        v_t = prev_state[:,3]
        steering_batch = torch.clamp(steering_batch, -0.5, 0.5)
        beta = steering_batch
        a_t = pedal_batch
        v_t_1 = v_t + a_t * dt 
        v_t_1 = torch.clamp(v_t_1, 0, 10)
        x_dot = v_t_1 * torch.cos(psi_t)
        y_dot = v_t_1 * torch.sin(psi_t)
        psi_dot = v_t * torch.tan(beta) / 2.5
        psi_dot = torch.clamp(psi_dot, -3.14 /2,3.14 /2)
        x_t_1 = x_dot * dt + x_t 
        y_t_1 = y_dot * dt + y_t
        psi_t_1 = psi_dot*dt + psi_t 
        current_state = torch.stack([x_t_1, y_t_1, psi_t_1, v_t_1], dim = 1)
        return current_state

    def decode(self, z, init_state):
        generated_traj = []
        prev_state = init_state 
        # decoder_input shape: batch_size x 4
        decoder_input = self.spatial_embedding(prev_state)
        decoder_input = decoder_input.view(1, -1 , self.embedding_dim)
        decoder_h = self.init_hidden_decoder(z)
        if len(decoder_h.shape) == 2:
            decoder_h = torch.unsqueeze(decoder_h, 0)
        decoder_h = (decoder_h, decoder_h)
        for _ in range(self.seq_len):
            # output shape: 1 x batch x h_dim
            output, decoder_h = self.decoder(decoder_input, decoder_h)
            control = self.hidden2control(output.view(-1, self.h_dim))
            curr_state = self.plant_model_batch(prev_state, control[:,0], control[:,1], self.dt)
            generated_traj.append(curr_state)
            decoder_input = self.spatial_embedding(curr_state)
            decoder_input = decoder_input.view(1, -1, self.embedding_dim)
            prev_state = curr_state 
        generated_traj = torch.stack(generated_traj, dim = 1)
        return generated_traj
    
    def forward(self, z, init_state):
        return self.decode(z, init_state)


class TrajVAE(nn.Module):
    def __init__(self,
        embedding_dim = 64,
        h_dim = 64,
        latent_dim = 100,
        seq_len = 30,
        use_relative_pos = True,
        dt = 0.03,
        kld_weight = 0.01,
        fde_weight = 0.1,
        ):
        super(TrajVAE, self).__init__()
        self.embedding_dim = embedding_dim
        self.h_dim = h_dim 
        self.num_layers = 1
        self.latent_dim = latent_dim
        self.seq_len = seq_len 
        self.use_relative_pos = use_relative_pos
        self.kld_weight = kld_weight
        self.fde_weight = fde_weight
        self.dt = dt
        self.vae_encoder = VaeEncoder(
            embedding_dim = self.embedding_dim,
            h_dim = self.h_dim,
            latemt_dim = self.latent_dim,
            seq_len = self.seq_len,
            use_relative_pos = self.use_relative_pos,
            dt = self.dt
        )
        self.vae_decoder = VaeDecoder(
            embedding_dim = self.embedding_dim,
            h_dim = self.h_dim,
            latemt_dim = self.latent_dim,
            seq_len = self.seq_len,
            use_relative_pos = self.use_relative_pos,
            dt = self.dt
        )

    def reparameterize(self, mu, logvar):
        # mu shape: batch size x latent_dim
        # sigma shape: batch_size x latent_dim
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def forward(self, expert_traj, init_state):
        mu, log_var = self.vae_encoder(expert_traj)
        z = self.reparameterize(mu, log_var)
        z = torch.tanh(z)
        recons_traj = self.vae_decoder(z, init_state)
        return [recons_traj, expert_traj, mu.squeeze(0), log_var.squeeze(0)]  

    def loss_function(self, *args):
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        epoch = 0
        if len(args) > 4:
            epoch = args[4]
        kld_weight = self.kld_weight
        recon_loss = 0
        # reconstruction loss
        recons_loss = F.mse_loss(recons[:,:,:2], input[:,:,:2])
        #recons_loss += F.mse_loss(recons[:,:,3], input[:,:,3]) * 0.01
        vel_loss = F.mse_loss(recons[:,:,3], input[:,:,3]) * 0.01
        #final displacement loss
        final_displacement_error = F.mse_loss(recons[:,-1, :2], input[:, -1, :2])
        theta_error = F.mse_loss(recons[:,:,2], input[:,:,2]) * 0.01
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        kld_weight = 0.01
        loss = recons_loss  + kld_weight * kld_loss + self.fde_weight * final_displacement_error + theta_error  + vel_loss
        return {'loss': loss, "reconstruction_loss": recons_loss, 'KLD': kld_loss, 'final_displacement_error' : final_displacement_error, 
        'theta_error':theta_error, 'mu':mu[0][0], 'log_var': log_var[0][0]}      


class WpDecoder(nn.Module):
    def __init__(self,
        control_num = 2,
        seq_len = 30,
        use_relative_pos = True,
        dt = 0.03,
        ):
        super(WpDecoder, self).__init__()
        self.control_num = control_num
        self.seq_len = seq_len 
        self.use_relative_pos = use_relative_pos
        self.dt = dt

    def plant_model_batch(self, prev_state_batch, pedal_batch, steering_batch, dt = 0.03):
        #import copy
        prev_state = prev_state_batch
        x_t = prev_state[:,0]
        y_t = prev_state[:,1]
        psi_t = prev_state[:,2]
        v_t = prev_state[:,3]
        steering_batch = steering_batch * 0.4
        steering_batch = torch.clamp(steering_batch, -0.5, 0.5)
        beta = steering_batch
        a_t = pedal_batch * 4
        v_t_1 = v_t + a_t * dt 
        v_t_1 = torch.clamp(v_t_1, 0, 10)
        psi_dot = v_t * torch.tan(beta) / 2.5
        psi_t_1 = psi_dot*dt + psi_t 
        x_dot = v_t_1 * torch.cos(psi_t_1)
        y_dot = v_t_1 * torch.sin(psi_t_1)
        x_t_1 = x_dot * dt + x_t 
        y_t_1 = y_dot * dt + y_t
        
        current_state = torch.stack([x_t_1, y_t_1, psi_t_1, v_t_1], dim = 1)
        return current_state

    def decode(self, z, init_state):
        generated_traj = []
        prev_state = init_state 
        assert z.shape[1] == self.seq_len * 2
        for i in range(self.seq_len):
            pedal_batch = z[:, 2*i]
            steer_batch = z[:, 2*i +1]
            curr_state = self.plant_model_batch(prev_state, pedal_batch, steer_batch, self.dt)
            generated_traj.append(curr_state)
            prev_state = curr_state 
        generated_traj = torch.stack(generated_traj, dim = 1)
        return generated_traj
    
    def forward(self, z, init_state):
        return self.decode(z, init_state)

class CCDecoder(nn.Module):
    def __init__(self,
        control_num = 2,
        seq_len = 30,
        use_relative_pos = True,
        dt = 0.03,
        ):
        super(CCDecoder, self).__init__()
        self.control_num = control_num
        self.seq_len = seq_len 
        self.use_relative_pos = use_relative_pos
        self.dt = dt

    def plant_model_batch(self, prev_state_batch, pedal_batch, steering_batch, dt = 0.03):
        prev_state = prev_state_batch
        x_t = prev_state[:,0]
        y_t = prev_state[:,1]
        psi_t = prev_state[:,2]
        v_t = prev_state[:,3]
        steering_batch = steering_batch * 0.4
        steering_batch = torch.clamp(steering_batch, -0.5, 0.5)
        beta = steering_batch
        a_t = pedal_batch * 4
        v_t_1 = v_t + a_t * dt 
        v_t_1 = torch.clamp(v_t_1, 0, 10)
        psi_dot = v_t * torch.tan(beta) / 2.5
        psi_t_1 = psi_dot*dt + psi_t 
        x_dot = v_t_1 * torch.cos(psi_t_1)
        y_dot = v_t_1 * torch.sin(psi_t_1)
        x_t_1 = x_dot * dt + x_t 
        y_t_1 = y_dot * dt + y_t
        
        current_state = torch.stack([x_t_1, y_t_1, psi_t_1, v_t_1], dim = 1)
        return current_state

    def decode(self, z, init_state):
        generated_traj = []
        prev_state = init_state 
        assert z.shape[1] == 2
        for i in range(self.seq_len):
            pedal_batch = z[:, 0]
            steer_batch = z[:, 1]
            curr_state = self.plant_model_batch(prev_state, pedal_batch, steer_batch, self.dt)
            generated_traj.append(curr_state)
            prev_state = curr_state 
        generated_traj = torch.stack(generated_traj, dim = 1)
        return generated_traj
    
    def forward(self, z, init_state):
        return self.decode(z, init_state)