import torch 

from data_utils import *
from model import *
from train import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dataset = CreditCard(root='./dataset', cos=0.5)

data = dataset[0]

N = data.x.shape[0]

model = DiscriminatorGAT(in_dim=N, out_dim= 32)
model.to(device)

optimzer = torch.optim.Adam(model.parameters())
loss_function = torch.nn.BCELoss()
