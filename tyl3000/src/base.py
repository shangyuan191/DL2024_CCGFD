from torch import nn
from modules import *
import modules
import torch

batch = 16
i_dim = 512
o_dim = 128
head = 4
x = torch.randn(batch, i_dim).to("cuda")

a = globa.simple_attention(i_dim, o_dim, head, True)

print(a)

class tab_model(nn.Module):
    def __init__(self):
        super(tab_model, self).__init__()
        pass

    def forward(self, x, A=None):
        #TODO:
        #   1:glob modules
        #   2:DGM(GSL)
        #   3:Downstream modules

        pass

