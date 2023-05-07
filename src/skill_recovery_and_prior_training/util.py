import math
import torch

def gaussian_probability(sigma, mu, regress_label):
    ONEOVERSQRT2PI = 1 / math.sqrt(2 * math.pi)
    probability = ONEOVERSQRT2PI / sigma * torch.exp(-0.5 * ((regress_label - mu) / sigma)**2)

    return probability

def nll_loss(mu, sigma, recovered_mu, sigma_weight=0.00):
    prob = gaussian_probability(sigma, mu, recovered_mu)
    nll = -torch.log(torch.clamp(torch.sum(prob, dim = 1), min = 1e-20))
    nll_loss = torch.mean(nll)

    # KL divergence to unit gaussian KL(p|unit)
    target_sigma = 0.61
    KL_loss = torch.mean(torch.sum(torch.log(target_sigma / sigma) + (sigma**2) / 2 / target_sigma**2 - 0.5, dim = 1))

    total_loss = nll_loss + sigma_weight * KL_loss

    return total_loss, nll_loss, KL_loss

def distribution_loss(mu, log_sigma, mu_label, log_sigma_label):
    sigma_label = torch.clamp(torch.exp(log_sigma_label), min = 1e-20)
    sigma = torch.clamp(torch.exp(log_sigma), min = 1e-20)

    KL_loss = torch.log(sigma_label / sigma) + (sigma**2 + (mu - mu_label)**2) / 2 / (sigma_label**2) - 0.5
    KL_loss = torch.mean(torch.sum(KL_loss, dim=1))

    return KL_loss

def action_error(predicted_logit_mu, predicted_logit_sigma, gt_logit_mu):
    def reparameterize(mu, sigma):
        # mu shape: batch size x latent_dim
        # sigma shape: batch_size x latent_dim
        # sigma = torch.exp(sigma)
        eps = torch.randn_like(sigma)
        return eps * sigma + mu

    # error of sampled logit 
    predicted_logit_mu_sampled = torch.tanh(reparameterize(predicted_logit_mu, predicted_logit_sigma))
    logit_mu1_sample_error_lst = abs(predicted_logit_mu_sampled - gt_logit_mu)[:,0].tolist()
    logit_mu2_sample_error_lst = abs(predicted_logit_mu_sampled - gt_logit_mu)[:,1].tolist()
    logit_mu3_sample_error_lst = abs(predicted_logit_mu_sampled - gt_logit_mu)[:,2].tolist()

    # error of mean logit 
    logit_mu1_mean_error_lst = abs(predicted_logit_mu - gt_logit_mu)[:,0].tolist()
    logit_mu2_mean_error_lst = abs(predicted_logit_mu - gt_logit_mu)[:,1].tolist()
    logit_mu3_mean_error_lst = abs(predicted_logit_mu - gt_logit_mu)[:,2].tolist()

    # error of mean latent variable 
    latent_var1_mean_error_lst = abs(torch.tanh(predicted_logit_mu) - torch.tanh(gt_logit_mu))[:,0]
    latent_var2_mean_error_lst = abs(torch.tanh(predicted_logit_mu) - torch.tanh(gt_logit_mu))[:,1]
    latent_var3_mean_error_lst = abs(torch.tanh(predicted_logit_mu) - torch.tanh(gt_logit_mu))[:,2]

    # error of mean skill params
    skill_param_lat1_mean_error_lst = abs(torch.tanh(predicted_logit_mu) - torch.tanh(gt_logit_mu))[:,0] * 5
    skill_param_yaw1_mean_error_lst = abs(torch.tanh(predicted_logit_mu) - torch.tanh(gt_logit_mu))[:,1] * 30
    skill_param_v1_mean_error_lst = abs(torch.tanh(predicted_logit_mu) - torch.tanh(gt_logit_mu))[:,2] * 5

    return logit_mu1_sample_error_lst, logit_mu2_sample_error_lst, logit_mu3_sample_error_lst, \
            logit_mu1_mean_error_lst, logit_mu2_mean_error_lst, logit_mu3_mean_error_lst, \
            latent_var1_mean_error_lst, latent_var2_mean_error_lst, latent_var3_mean_error_lst, \
            skill_param_lat1_mean_error_lst, skill_param_yaw1_mean_error_lst, skill_param_v1_mean_error_lst

