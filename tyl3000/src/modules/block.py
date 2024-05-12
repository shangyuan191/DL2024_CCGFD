import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch.nn import Linear, ReLU, Dropout
from torch_geometric.nn import SAGEConv, GATConv, GINEConv, GCNConv, Sequential, summary
from torch_geometric.utils import *
import scipy.sparse as sp

import gen
import globa

class base_block(nn.Module):
    def __init__(self):
        super().__init__()

class dgm_block(base_block):
    # 20240512v1 test block
    def __init__(self, i_dim, h_dim, o_dim, head):
        super().__init__()
        d_DGM = gen.dDGM
        att = globa.simple_attention
        mpnn_model = GCNConv

        self.block = Sequential('x, edge_index',
            [
                (d_DGM(att(i_dim, h_dim, head)), 'x, edge_index -> x, edge_index, edge_att'),
                (ReLU(inplace=True), 'x -> x'),

                (mpnn_model(h_dim, h_dim), 'x, edge_index -> x1'),
                (ReLU(inplace=True), 'x1 -> x1'),
                (self.res, 'x, x1 -> x'),


                (d_DGM(mpnn_model(h_dim, h_dim)), 'x, edge_index -> x1, edge_index, edge_att'),
                (ReLU(inplace=True), 'x1 -> x1'),
                (self.res, 'x, x1 -> x'),
                (mpnn_model(h_dim, h_dim), 'x, edge_index -> x1'),
                (ReLU(inplace=True), 'x1 -> x1'),
                (self.res, 'x, x1 -> x'),

                (d_DGM(mpnn_model(h_dim, h_dim)), 'x, edge_index -> x1, edge_index, edge_att'),
                (ReLU(inplace=True), 'x1 -> x1'),
                (self.res, 'x, x1 -> x'),
                (mpnn_model(h_dim, o_dim), 'x, edge_index -> x'),
            ]
        )

        self.a1 = d_DGM(att(i_dim, h_dim, head))
        self.a2 = mpnn_model(h_dim, o_dim)

    def res(self, x1, x2):
        return x1 + x2


    def forward(self, x, edge_index=None):
        x = self.block(x, edge_index)
        return x


if __name__ == '__main__':
    batch = 3000
    i_dim = 512
    h_dim = 128
    o_dim = 64
    head = 4
    x = torch.randn(batch, i_dim).to("cuda")
    edge_index = None

    model = dgm_block(i_dim, h_dim, o_dim, head).to("cuda")
    x = model(x, edge_index)
    print(x.shape)