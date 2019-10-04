import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.distributions import Normal, Categorical

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

def set_seed(n_seed):
    random.seed(n_seed)
    np.random.seed(n_seed)
    torch.manual_seed(n_seed)
    if device == "cuda":
        torch.cuda.manual_seed(n_seed)

def gaussian_likelihood(x, mu, log_std, EPS):
    likelihood = -0.5 * (((x-mu)/(torch.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
    return likelihood

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

def atanh(y):
    return 0.5*(torch.log((1+y+1e-12)/(1-y+1e-12)))

###########################################################################
#
#                               Classes
#
###########################################################################

class QNetwork(nn.Module):
    def __init__(self,
                n_actions,
                input_width=84,
                channel_sizes=[4,16,32],
                kernel_sizes=[8,4],
                stride_sizes=[4,2],
                fully_connected_sizes=[256],
                lr=2.5e-4):
        super().__init__()

        self.l1 = nn.Conv2d(channel_sizes[0], channel_sizes[1], kernel_sizes[0], stride=stride_sizes[0])
        self.l2 = nn.Conv2d(channel_sizes[1], channel_sizes[2], kernel_sizes[1], stride=stride_sizes[1])
        output_width_l1 = output_width(input_width, kernel_sizes[0], stride_sizes[0])
        output_width_l2 = output_width(output_width_l1, kernel_sizes[1], stride_sizes[1])
        self.l3 = nn.Linear(channel_sizes[2]*output_width_l2**2, fully_connected_sizes[0])
        self.l4 = nn.Linear(fully_connected_sizes[0], n_actions)

        self.apply(weights_init_)

        self.loss_func = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, observation):
        x = F.relu(self.l1(observation))
        x = F.relu(self.l2(x))
        x = x.view(observation.size(0),-1)
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return x