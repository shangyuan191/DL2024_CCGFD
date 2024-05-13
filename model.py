from collections import OrderedDict

import torch
import torch.nn as nn

from torch_geometric.nn import GATConv


class GATEncoder(nn.Module):

    def __init__(self, in_dim, out_dim, num_heads, n_layer, hidden_dim):

        super(GATEncoder, self).__init__()

        self.agg = nn.ModuleList()
        self.n_layer = n_layer

        for i in range(n_layer):
            if i ==0 :                      # first layer
                conv = GATConv(in_dim, hidden_dim, heads=num_heads, add_self_loops=False)
            elif i != n_layer - 1:          # middle layer
                conv = GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, add_self_loops=False)
            else:                           # last layer
                conv = GATConv(hidden_dim * num_heads, out_dim, num_heads, concat=False, add_self_loops=False)

            self.agg.append(conv)

    def forward(self, x, edge_index):

        for m in self.agg:
            x = m(x, edge_index)

        return x


class DiscriminatorGAT(nn.Module):

    def __init__(self, in_dim, out_dim, num_heads, n_layer, hidden_dim):
        
        super(DiscriminatorGAT, self).__init__()

        self.ego_lin = nn.Sequential(OrderedDict([
                                                ('ego_1', nn.Linear(in_dim, 64)), 
                                                ('ego_2', nn.Linear(64, out_dim))
                                                ])
                                    )
        
        self.gconv = GATEncoder(in_dim, out_dim, num_heads, n_layer, hidden_dim)

    def forward(self, x, edge_index):

        mlp = self.ego_lin(x)

        gat = self.gconv(x, edge_index)

        return mlp, gat
    