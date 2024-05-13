import yaml

import torch 
import torch.nn.functional as F


def rescale_cosine_sim(source, target):

    score = F.cosine_similarity(source, target)

    return (-1 * score.unsqueeze(0) + 1) / 2 + 1e-06



def update_argparse(args, path):

    with open(path, 'r') as yamlfile:
        config = yaml.safe_load(yamlfile)

    args.update(config)

    return args