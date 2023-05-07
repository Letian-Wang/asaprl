import torch, os, glob
import pdb, argparse
from tqdm import tqdm
from asaprl.policy.conv_qac import ConvQAC
from ding.policy import SACPolicy
from util import nll_loss, action_error, distribution_loss
from dataset import create_raw_data_loader, create_annotated_data_loader
from parameter import hyper_parameter
from tensorboardX import SummaryWriter
from torch import optim
from torch.optim.lr_scheduler import StepLR

parser = argparse.ArgumentParser()
parser.add_argument('--annotate_skill', type=bool, default=False)
parser.add_argument('--KL_weight', type=float, default=1)
parser.add_argument('--action_shape', type=int, default=3)
parser.add_argument('--scenario', type=str, default='highway')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--exp_name', type=str, default='test')
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--n_epochs', type=int, default=1000)
args = parser.parse_args()
params = hyper_parameter(args)
params.mkdir_write_params_gt()
print("device: ", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

''' model '''
# config
model_config = dict(
            obs_shape=[5, 200, 200],
            action_shape=params.action_shape,
            encoder_hidden_size_list=[128, 128, 64],
)
model_config2 = SACPolicy.config['model']
model_config2.update(model_config)
model = ConvQAC(**model_config2)

''' data '''
print("params.scenario, params.batch_size: ", params.scenario, params.batch_size)
if params.annotate_skill:
    train_loader, val_loader = create_annotated_data_loader(params.scenario, params.batch_size)
else:
    train_loader, val_loader = create_raw_data_loader(params.scenario, params.batch_size)
''' training '''
tb_logger = SummaryWriter('saved_model/pretrain_actor_gt_skill/log/{}/'.format(params.exp_name))
optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
scheduler = StepLR(optimizer, step_size=100, gamma=0.5, verbose=True)

model = model.to(params.device)
best_val_loss = 1000
for epoch in range(params.n_epochs):
    decs = 'Train - epoch-{}'.format(epoch)
    model.train()
    train_loss_one_epoch = 0
    train_nll_loss_one_epoch = 0
    train_KL_loss_one_epoch = 0
    train_gt_logit_mu1_sample_error_one_epoch, train_gt_logit_mu2_sample_error_one_epoch, train_gt_logit_mu3_sample_error_one_epoch= [], [], []
    train_gt_logit_mu1_mean_error_one_epoch, train_gt_logit_mu2_mean_error_one_epoch, train_gt_logit_mu3_mean_error_one_epoch= [], [], []
    train_gt_latent_var1_mean_error_one_epoch, train_gt_latent_var2_mean_error_one_epoch, train_gt_latent_var3_mean_error_one_epoch = [], [], []
    train_gt_skill_param_lat1_mean_error_one_epoch, train_gt_skill_param_yaw1_mean_error_one_epoch, train_gt_skill_param_v1_mean_error_one_epoch = [], [], []

    # parameter relationship: logit (output by the policy, [-∞, ∞]) ----tanh()---->  latent variable [-1, 1] ----transform()---->  motion skll param
    for train_obs, train_gt_latent_var, train_gt_logit in tqdm(train_loader, desc = decs):
        train_obs = train_obs.float().to(params.device)
        train_gt_latent_var = train_gt_latent_var.float().to(params.device)                   # latent variable output by RL agent
        train_gt_logit_mu = torch.arctanh(train_gt_latent_var)                                # logit output by RL agent
        predict_action = model.forward(train_obs, 'compute_actor')                      # predicted logit
        predicted_logit_mu, predicted_logit_sigma = predict_action['logit'][0], predict_action['logit'][1]  
        
        # loss
        train_total_loss, train_nll_loss, train_KL_loss = nll_loss(predicted_logit_mu, predicted_logit_sigma, train_gt_logit_mu, params.KL_weight)
        train_loss = train_total_loss

        # error of predicted logits w.r.t ground-truth logits from RL expert
        logit_mu1_sample_error_lst, logit_mu2_sample_error_lst, logit_mu3_sample_error_lst, \
        logit_mu1_mean_error_lst, logit_mu2_mean_error_lst, logit_mu3_mean_error_lst, \
        latent_var1_mean_error_lst, latent_var2_mean_error_lst, latent_var3_mean_error_lst, \
        skill_param_lat1_mean_error_lst, skill_param_yaw1_mean_error_lst, skill_param_v1_mean_error_lst  = action_error(predicted_logit_mu, predicted_logit_sigma, train_gt_logit_mu)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # record loss
        train_loss_one_epoch += train_loss.item()
        train_nll_loss_one_epoch += train_nll_loss.item()
        train_KL_loss_one_epoch += train_KL_loss.item()
        # record error of predicted logits w.r.t logits from RL expert
            # sampled logit
        train_gt_logit_mu1_sample_error_one_epoch += logit_mu1_sample_error_lst
        train_gt_logit_mu2_sample_error_one_epoch += logit_mu2_sample_error_lst
        train_gt_logit_mu3_sample_error_one_epoch += logit_mu3_sample_error_lst
            # mean logit
        train_gt_logit_mu1_mean_error_one_epoch += logit_mu1_mean_error_lst
        train_gt_logit_mu2_mean_error_one_epoch += logit_mu2_mean_error_lst
        train_gt_logit_mu3_mean_error_one_epoch += logit_mu3_mean_error_lst
            # mean latent variable
        train_gt_latent_var1_mean_error_one_epoch += latent_var1_mean_error_lst
        train_gt_latent_var2_mean_error_one_epoch += latent_var2_mean_error_lst
        train_gt_latent_var3_mean_error_one_epoch += latent_var3_mean_error_lst
            # mean skill parameter
        train_gt_skill_param_lat1_mean_error_one_epoch += skill_param_lat1_mean_error_lst
        train_gt_skill_param_yaw1_mean_error_one_epoch += skill_param_yaw1_mean_error_lst
        train_gt_skill_param_v1_mean_error_one_epoch += skill_param_v1_mean_error_lst

    print("training loss: ", train_loss_one_epoch)
    list_mean = lambda lst: sum(lst) / len(lst)
    tb_logger.add_scalar("train/loss_epoch", train_loss_one_epoch, epoch)
    tb_logger.add_scalar("train/nll_loss_epoch", train_nll_loss_one_epoch, epoch)
    tb_logger.add_scalar("train/KL_loss_epoch", train_KL_loss_one_epoch, epoch)
    tb_logger.add_scalar("train/gt_logit_mu1_sample_error_one_epoch", list_mean(train_gt_logit_mu1_sample_error_one_epoch), epoch)
    tb_logger.add_scalar("train/gt_logit_mu2_sample_error_one_epoch", list_mean(train_gt_logit_mu2_sample_error_one_epoch), epoch)
    tb_logger.add_scalar("train/gt_logit_mu3_sample_error_one_epoch", list_mean(train_gt_logit_mu3_sample_error_one_epoch), epoch)
    tb_logger.add_scalar("train/gt_logit_mu1_mean_error_one_epoch", list_mean(train_gt_logit_mu1_mean_error_one_epoch), epoch)
    tb_logger.add_scalar("train/gt_logit_mu2_mean_error_one_epoch", list_mean(train_gt_logit_mu2_mean_error_one_epoch), epoch)
    tb_logger.add_scalar("train/gt_logit_mu3_mean_error_one_epoch", list_mean(train_gt_logit_mu3_mean_error_one_epoch), epoch)
    tb_logger.add_scalar("train/gt_latent_var1_mean_error_one_epoch", list_mean(train_gt_latent_var1_mean_error_one_epoch), epoch)
    tb_logger.add_scalar("train/gt_latent_var2_mean_error_one_epoch", list_mean(train_gt_latent_var2_mean_error_one_epoch), epoch)
    tb_logger.add_scalar("train/gt_latent_var3_mean_error_one_epoch", list_mean(train_gt_latent_var3_mean_error_one_epoch), epoch)
    tb_logger.add_scalar("train/gt_skill_param_lat1_mean_error_one_epoch", list_mean(train_gt_skill_param_lat1_mean_error_one_epoch), epoch)
    tb_logger.add_scalar("train/gt_skill_param_yaw1_mean_error_one_epoch", list_mean(train_gt_skill_param_yaw1_mean_error_one_epoch), epoch)
    tb_logger.add_scalar("train/gt_skill_param_v1_mean_error_one_epoch", list_mean(train_gt_skill_param_v1_mean_error_one_epoch), epoch)


    if epoch > 0 and epoch % params.val_freq == 0:
        model.eval()
        val_loss_one_epoch = 0
        val_nll_loss_one_epoch = 0
        val_KL_loss_one_epoch = 0

        val_gt_logit_mu1_sample_error_one_epoch, val_gt_logit_mu2_sample_error_one_epoch, val_gt_logit_mu3_sample_error_one_epoch= [], [], []
        val_gt_logit_mu1_mean_error_one_epoch, val_gt_logit_mu2_mean_error_one_epoch, val_gt_logit_mu3_mean_error_one_epoch= [], [], []
        val_gt_latent_var1_mean_error_one_epoch, val_gt_latent_var2_mean_error_one_epoch, val_gt_latent_var3_mean_error_one_epoch = [], [], []
        val_gt_skill_param_lat1_mean_error_one_epoch, val_gt_skill_param_yaw1_mean_error_one_epoch, val_gt_skill_param_v1_mean_error_one_epoch = [], [], []

        for val_obs, val_gt_latent_var, val_gt_logit in tqdm(val_loader, desc = 'Val  '):
            with torch.no_grad():
                val_obs = val_obs.float().to(params.device)
                val_gt_latent_var = val_gt_latent_var.float().to(params.device)                   # latent variable output by RL agent
                val_gt_logit_mu = torch.arctanh(val_gt_latent_var)                                # logit output by RL agent
                predict_action = model.forward(val_obs, 'compute_actor')                      # predicted logit
                predicted_logit_mu, predicted_logit_sigma = predict_action['logit'][0], predict_action['logit'][1]  
                
                # loss
                val_total_loss, val_nll_loss, val_KL_loss = nll_loss(predicted_logit_mu, predicted_logit_sigma, val_gt_logit_mu, params.KL_weight)
                val_loss = val_total_loss

                # error of predicted logits w.r.t ground-truth logits from RL expert
                logit_mu1_sample_error_lst, logit_mu2_sample_error_lst, logit_mu3_sample_error_lst, \
                logit_mu1_mean_error_lst, logit_mu2_mean_error_lst, logit_mu3_mean_error_lst, \
                latent_var1_mean_error_lst, latent_var2_mean_error_lst, latent_var3_mean_error_lst, \
                skill_param_lat1_mean_error_lst, skill_param_yaw1_mean_error_lst, skill_param_v1_mean_error_lst  = action_error(predicted_logit_mu, predicted_logit_sigma, val_gt_logit_mu)

                # record loss
                val_loss_one_epoch += val_loss.item()
                val_nll_loss_one_epoch += val_nll_loss.item()
                val_KL_loss_one_epoch += val_KL_loss.item()
                # record error of predicted logits w.r.t logits from RL expert
                    # sampled logit
                val_gt_logit_mu1_sample_error_one_epoch += logit_mu1_sample_error_lst
                val_gt_logit_mu2_sample_error_one_epoch += logit_mu2_sample_error_lst
                val_gt_logit_mu3_sample_error_one_epoch += logit_mu3_sample_error_lst
                    # mean logit
                val_gt_logit_mu1_mean_error_one_epoch += logit_mu1_mean_error_lst
                val_gt_logit_mu2_mean_error_one_epoch += logit_mu2_mean_error_lst
                val_gt_logit_mu3_mean_error_one_epoch += logit_mu3_mean_error_lst
                    # mean latent variable
                val_gt_latent_var1_mean_error_one_epoch += latent_var1_mean_error_lst
                val_gt_latent_var2_mean_error_one_epoch += latent_var2_mean_error_lst
                val_gt_latent_var3_mean_error_one_epoch += latent_var3_mean_error_lst
                    # mean skill parameter
                val_gt_skill_param_lat1_mean_error_one_epoch += skill_param_lat1_mean_error_lst
                val_gt_skill_param_yaw1_mean_error_one_epoch += skill_param_yaw1_mean_error_lst
                val_gt_skill_param_v1_mean_error_one_epoch += skill_param_v1_mean_error_lst

        print("val loss: ", val_loss_one_epoch)
        list_mean = lambda lst: sum(lst) / len(lst)
        tb_logger.add_scalar("val/loss_epoch", val_loss_one_epoch, epoch)
        tb_logger.add_scalar("val/nll_loss_epoch", val_nll_loss_one_epoch, epoch)
        tb_logger.add_scalar("val/KL_loss_epoch", val_KL_loss_one_epoch, epoch)
        tb_logger.add_scalar("val/gt_logit_mu1_sample_error_one_epoch", list_mean(val_gt_logit_mu1_sample_error_one_epoch), epoch)
        tb_logger.add_scalar("val/gt_logit_mu2_sample_error_one_epoch", list_mean(val_gt_logit_mu2_sample_error_one_epoch), epoch)
        tb_logger.add_scalar("val/gt_logit_mu3_sample_error_one_epoch", list_mean(val_gt_logit_mu3_sample_error_one_epoch), epoch)
        tb_logger.add_scalar("val/gt_logit_mu1_mean_error_one_epoch", list_mean(val_gt_logit_mu1_mean_error_one_epoch), epoch)
        tb_logger.add_scalar("val/gt_logit_mu2_mean_error_one_epoch", list_mean(val_gt_logit_mu2_mean_error_one_epoch), epoch)
        tb_logger.add_scalar("val/gt_logit_mu3_mean_error_one_epoch", list_mean(val_gt_logit_mu3_mean_error_one_epoch), epoch)
        tb_logger.add_scalar("val/gt_latent_var1_mean_error_one_epoch", list_mean(val_gt_latent_var1_mean_error_one_epoch), epoch)
        tb_logger.add_scalar("val/gt_latent_var2_mean_error_one_epoch", list_mean(val_gt_latent_var2_mean_error_one_epoch), epoch)
        tb_logger.add_scalar("val/gt_latent_var3_mean_error_one_epoch", list_mean(val_gt_latent_var3_mean_error_one_epoch), epoch)
        tb_logger.add_scalar("val/gt_skill_param_lat1_mean_error_one_epoch", list_mean(val_gt_skill_param_lat1_mean_error_one_epoch), epoch)
        tb_logger.add_scalar("val/gt_skill_param_yaw1_mean_error_one_epoch", list_mean(val_gt_skill_param_yaw1_mean_error_one_epoch), epoch)
        tb_logger.add_scalar("val/gt_skill_param_v1_mean_error_one_epoch", list_mean(val_gt_skill_param_v1_mean_error_one_epoch), epoch)

    if epoch > 0 and epoch % params.save_freq == 0 and val_loss_one_epoch < best_val_loss:
        best_val_loss = val_loss_one_epoch
        # delete previous ckpt in the directory
        for f in glob.glob("./saved_model/pretrain_actor_gt_skill/{}/ckpt/*".format(params.exp_name)):
            os.remove(f)
        torch.save(model.state_dict(), "./saved_model/pretrain_actor_gt_skill/{}/ckpt/{}_model_ckpt".format(params.exp_name, epoch))   
        torch.save(model.actor.state_dict(), "./saved_model/pretrain_actor_gt_skill/{}/ckpt/{}_actor_ckpt".format(params.exp_name, epoch))

    scheduler.step()
