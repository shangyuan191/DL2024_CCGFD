import os.path as osp
import torch

from tqdm import tqdm
from utils import *

def train(args, data, model, optim, loss_function, device):

    stats = {
        "best_loss": 1e9,
        "best_epoch": -1,
    }
    
    state_path = osp.join(args['ckpt_path'], f"creaditcard_cos{int(args['cos']*10)}.pkl")
    batch_size = args['batch_size']

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)

    N = x.shape[0]

    model.train()

    for epoch in tqdm(range(args['total_epoch'])):

        optim.zero_grad()

        label_zeros = torch.zeros([1, N]).to(device)
        label_ones = torch.ones([1, N]).to(device)

        x_mlp, x_gat = model(x, edge_index)
        idx = torch.randperm(len(x_mlp))
        x_mlp_permu = x_mlp[idx]
        x_gat_permu = x_gat[idx]

        score_pos = rescale_cosine_sim(x_mlp, x_gat)
        score_nn = rescale_cosine_sim(x_mlp, x_gat_permu)
        score_en = rescale_cosine_sim(x_mlp, x_mlp_permu)

        loss_pos = loss_function(score_pos, label_zeros)
        loss_aug = loss_function(score_nn, label_ones)
        loss_nod = loss_function(score_en, label_ones)

        loss_sum = loss_pos \
                + args['alpha'] * loss_aug \
                + args['gamma'] * loss_nod
        
        # Rescale loss
        loss_sum.backward()

        if loss_pos < stats["best_loss"]:
            stats["best_loss"] = loss_pos.item()
            stats["best_epoch"] = epoch
            torch.save(model.state_dict(), state_path)
        optim.step()
        
    return stats
