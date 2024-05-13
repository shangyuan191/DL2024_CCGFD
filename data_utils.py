import numpy as np
import torch
import os.path as osp

from torch_geometric.data import Data, InMemoryDataset

class CreditCard(InMemoryDataset):

    def __init__(self, root, cos, transform=None, pre_transform=None):

        self.name = f'creaditcard_cos{int(cos*10)}'
        self.cos = cos

        super().__init__(root, transform, pre_transform)
        
        self.load(self.processed_paths[0])


    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return ['creditcard.npz']

    @property
    def processed_file_names(self):
        return ['data.pt']
    
    def download(self):
        pass

    def process(self):
        
        data = np.load(self.raw_paths[0])

        x = torch.from_numpy(data['features']).to(torch.float32)
        y = torch.from_numpy(data['label'])
        cos_sim = data['cosine_sim']

        mask = cos_sim > self.cos
        src, dst = np.nonzero(mask)
        edge_index = torch.from_numpy(np.stack([src, dst]))

        data = Data(x=x, y=y, edge_index=edge_index)
        self.save([data], self.processed_paths[0])