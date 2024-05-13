from collections import OrderedDict

import torch
import torch.nn as nn

from torch_geometric.nn import GATConv


class DiscriminatorGAT(nn.Module):

    def __init__(self, in_dim, out_dim):
        
        super().__init__()

        self.ego_lin = nn.Sequential(OrderedDict([
                                                ('ego_1', nn.Linear(in_dim, 64)), 
                                                ('ego_2', nn.Linear(64, out_dim))
                                                ])
                                    )
        
        self.gconv = nn.ModuleList([GATConv(in_dim, 64), 
                                GATConv(64, out_dim)]
                                    )

    def forward(self, x, edge_index):

        source = self.ego_lin(x)

        for m in self.gconv:
            x = m(x, edge_index)

        return source, x
    