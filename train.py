import torch

from random import shuffle

from utils import *

def train(args, data, model, optim, loss_function):

    stats = {
        "best_loss": 1e9,
        "best_epoch": -1,
    }
    state_path = args.ckpt_path

    batch_size = args.batch_size
    device = args.device

    x = data.x
    edge_index = data.edge_index

    N = x.shape[0]

    model.train()

    for epoch in range(args.num_epoch):

        optim.zero_grad()

        i = 0 
        loss_pos = 0
        while i * batch_size < N:

            start_index = i * batch_size
            end_index = min((i + 1) * batch_size, N)
            i += 1

            x_batch = x[start_index:edge_index, :]

            label_zeros = torch.zeros([1,len(x_batch)]).to(device)
            label_ones = torch.ones([1,len(x_batch)]).to(device)

            x_mlp, x_gat = model(x_batch, edge_index)
            idx = torch.randperm(len(x_mlp))
            x_mlp_permu = x_mlp[idx]
            x_gat_permu = x_gat[idx]

            score_pos = rescale_cosine_sim(x_mlp, x_gat)
            score_nn = rescale_cosine_sim(x_mlp, x_gat_permu)
            score_en = rescale_cosine_sim(x_mlp, x_mlp_permu)

            loss_pos_batch = loss_function(score_pos, label_zeros)
            loss_aug_batch = loss_function(score_nn, label_ones)
            loss_nod_batch = loss_function(score_en, label_ones)

            loss_sum_batch = loss_pos_batch \
                    + args.alpha * loss_aug_batch \
                    + args.gamma * loss_nod_batch
            
            # Rescale loss
            loss_sum_batch = loss_sum_batch * (end_index - start_index) / N
            loss_sum_batch.backward()
            loss_pos += loss_pos_batch.item()

        if loss_pos < stats["best_loss"]:
            stats["best_loss"] = loss_pos
            stats["best_epoch"] = epoch
            torch.save(model.state_dict(), state_path)
        optim.step()
        
    return stats
