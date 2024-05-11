from collections import OrderedDict

import torch
import torch.nn as nn

from torch_geometric.nn import GATConv


class DiscriminatorGAT(nn.Module):

    def __init__(self, in_dim, out_dim):
        
        self.ego_lin = nn.Sequential(OrderedDict[('ego_1', nn.Linear(in_dim, 64)), ('ego_2', nn.Linear(64, out_dim))])
        self.gconv = nn.Sequential(OrderedDict[('neigh_1', GATConv(in_dim, 64)), ('neigh_2', GATConv(64, out_dim))])

    def forward(self, source, terget):

        sorce = nn.functional.cosine_similarity(source, terget)

        return -1 * sorce.unsqueeze(0)
    