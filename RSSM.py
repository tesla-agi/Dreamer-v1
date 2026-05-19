from GRU import GRU
import torch
import torch.nn as nn
import torch.nn.functional as F

class RSSM(nn.Module):
    def __init__(self,s_dim=32,a_dim=1,hidden_dim=256,emb_dim=200,min_std=0.1):
        super(RSSM,self).__init__()
        self.s_dim=s_dim
        self.hidden_dim=hidden_dim
        self.min_std=min_std

        self.gru=GRU(input_dim=s_dim+a_dim,hidden_dim=hidden_dim)

        self.prior_net=nn.Sequential(
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,2*s_dim),
        )
        self.posterior_net=nn.Sequential(
            nn.Linear(emb_dim+hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,2*s_dim),
        )

