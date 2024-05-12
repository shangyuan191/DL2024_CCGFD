import yaml
from sklearn.metrics import roc_auc_score
import os
import argparse
import numpy as np
import torch

from tqdm import tqdm

from utils import update_argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GGD Anomaly')

    args = vars(parser.parse_args())
    args = update_argparse(args, './config.yaml')

    # Setup torch
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Load dataset
    g, features, ano_label, _, _ = load_dataset(args.dataset)
    features = torch.FloatTensor(features)
    if args.batch_size == -1:
        features = features.to(device)
    g = g.to(device)
    dataloader = Dataloader(g, features, args.k, dataset_name=args.dataset)
    if not os.path.isdir("./ckpt"):
        os.makedirs("./ckpt")

    # Run the experiment
    seed = args.seed
    model, stats = run_experiment(args, seed, device, dataloader, ano_label)
    print("AUC: %.4f" % stats["AUC"])
    print("Time (Train): %.4fs" % stats["time_train"])
    print("Mem (Train): %.4f MB" % (stats["mem_train"] / 1024 / 1024))
    print("Time (Test): %.4fs" % stats["time_test"])
    print("Mem (Test): %.4f MB" % (stats["mem_test"] / 1024 / 1024))
    exit()