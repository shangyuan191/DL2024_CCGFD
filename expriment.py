import os
import torch 
import argparse

from data_utils import *
from utils import *
from model import *
from train import *


parser = argparse.ArgumentParser(description="PREM-GAT")

parser.add_argument('--cos', type=float, default=0.5)
parser.add_argument('--config_path', type= str, default='./config.yaml')

    
if __name__ == "__main__":

    args = vars(parser.parse_args())
    
    # load config
    args = update_argparse(args, args['config_path'])

    device = torch.device('cuda:0' if torch.cuda.is_available() else'cpu')
    os.makedirs(args['ckpt_path'], exist_ok=True)


    dataset = CreditCard(root='./dataset', cos=args['cos'])
    data = dataset[0]

    N = data.x.shape[0]
    num_feature = data.x.shape[1]

    model = DiscriminatorGAT(in_dim=num_feature, out_dim= 32)
    model.to(device)

    optimzer = torch.optim.Adam(model.parameters())
    loss_function = torch.nn.BCELoss()

    stat = train(args, data, model, optimzer, loss_function,device)

    print(stat)