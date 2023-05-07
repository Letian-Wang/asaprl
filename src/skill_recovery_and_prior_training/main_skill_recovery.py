import torch
import pdb, argparse
from skill_param_recovery import annotate_data
from tensorboardX import SummaryWriter
from torch import optim
from parameter import hyper_parameter

parser = argparse.ArgumentParser()
parser.add_argument('--annotate_skill', type=bool, default=False)
parser.add_argument('--KL_weight', type=float, default=0.0)
parser.add_argument('--action_shape', type=int, default=3)
parser.add_argument('--scenario', type=str, default='highway')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--exp_name', type=str, default='test')
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--n_epochs', type=int, default=100)
print("device: ", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
args = parser.parse_args()
params = hyper_parameter(args)
params.mkdir_write_params()

annotate_data(params.scenario, params.batch_size)