import math
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, data_dim, hidden_dim):
        super(Generator,self).__init__()
        self.net = nn.Sequential(
            nn.Linear(data_dim,hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim,hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim,data_dim),
        )
    def forward(self,z):
        output = self.net(z)
        return output

class Discriminator(nn.Module):
    def __init__(self, data_dim, hidden_dim):
        super(Discriminator,self).__init__()
        self.net=nn.Sequential(
            nn.Linear(data_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    def forward(self,x):
        output = self.net(x)
        return output
