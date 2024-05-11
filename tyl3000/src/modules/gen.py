import torch
import torch.nn as nn
import torch_geometric

class base_edage:
    def __init__(self):
        super().__init__()
        pass
    def pairwise_euclidean_distances(self, x, dim=-1):
        dist = torch.cdist(x, x) ** 2
        return dist

    def pairwise_poincare_distances(self, x, dim=-1):
        x_norm = (x ** 2).sum(dim, keepdim=True)
        x_norm = (x_norm.sqrt() - 1).relu() + 1
        x = x / (x_norm * (1 + 1e-2))
        x_norm = (x ** 2).sum(dim, keepdim=True)

        pq = torch.cdist(x, x) ** 2
        dist = torch.arccosh(1e-6 + 1 + 2 * pq / ((1 - x_norm) * (1 - x_norm.transpose(-1, -2)))) ** 2
        return dist

    def forward(self, x, A_0=None):
        pass
    def reset_parameters(self):
        pass

class cDGM(base_edage):
    #   20240511_v1 done
    def __init__(self, embed_model, k=None, distance="euclidean"):
        super(cDGM, self).__init__()
        _input_dim = 4
        self.embed_model = embed_model

        self.T = nn.Parameter(torch.tensor(1.))
        self.t = nn.Parameter(torch.tensor(0.5))

        self.scale = nn.Parameter(torch.tensor(-1).float(),requires_grad=False)
        self.centroid = nn.Parameter(torch.zeros((1,1,_input_dim)).float(),requires_grad=False)

    def forward(self, x, A_0=None):
        x = self.embed_model(x, A_0)

        if self.scale < 0:
            self.centroid.data = x.mean(-2, keepdim=True).detach()
            self.scale.data = (0.9 / (x - self.centroid).abs().max()).detach()

        D = self.distance((x - self.centroid) * self.scale)
        A = torch.sigmoid(self.t * (self.T.abs() - D))

        return x, A, None


class dDGM(base_edage):
    #   20240511_v1 done
    def __init__(self, embed_model, k=5, distance="euclidean", sparse=True):
        super(dDGM, self).__init__()
        self.embed_model = embed_model
        self.k = k

        self.temperature = nn.Parameter(torch.tensor(1.).float())

        if distance == 'euclidean':
            self.distance = self.pairwise_euclidean_distances
        else:
            self.distance = self.pairwise_poincare_distances

    def forward(self, x, A_0=None):
        x = self.embed_model(x, A_0)
        with torch.set_grad_enabled(self.training):
            D = self.distance(x)
            edges_hat, logprobs = self.sample_without_replacement(D)

        return x, edges_hat, logprobs

if __name__ == '__main__':

