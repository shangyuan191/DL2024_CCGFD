import torch
import torch.nn as nn
import torch_geometric

class base_edage:
    def __init__(self):
        super().__init__()
        pass
    def pairwise_euclidean_distances(self, x, dim=-1):
        dist = torch.cdist(x, x) ** 2
        return dist, x

    def pairwise_poincare_distances(self, x, dim=-1):
        x_norm = (x ** 2).sum(dim, keepdim=True)
        x_norm = (x_norm.sqrt() - 1).relu() + 1
        x = x / (x_norm * (1 + 1e-2))
        x_norm = (x ** 2).sum(dim, keepdim=True)

        pq = torch.cdist(x, x) ** 2
        dist = torch.arccosh(1e-6 + 1 + 2 * pq / ((1 - x_norm) * (1 - x_norm.transpose(-1, -2)))) ** 2
        return dist, x

    def forward(self, x, A_0=None):
        pass
    def reset_parameters(self):
        pass

class cDGM(base_edage):
    # not done yet
    def __init__(self, d_in):
        super(cDGM, self).__init__()

        #self.gcn_conv = GCNConv(d_in, 32)
        #self.mlp = MLP([d_in, 32], final_activation=True)

        # let t be a learnable parameter
        self.t = nn.Parameter(torch.tensor(0.05))
        self.T = nn.Parameter(torch.tensor(1.))



    def forward(self, X, A_0= None, fix_DGM=False):

        # if an input graph G_0 is provided as adjacency matrix A_0
        if A_0 is not None and not fix_DGM:

            # apply edge or graph convolution
            X_hat = self.gcn_conv(X, A_0)

        else:

            # apply the multi-layer perceptron
            X_hat = self.mlp(X)

        # Initialise D (instead of P!) as a matrix of zeros
        D = torch.zeros(X_hat.shape[0], X_hat.shape[0], dtype=torch.float)

        # For every combination of nodes

        A = torch.sigmoid(self.t * (self.T - D))
        return A


class dDGM(base_edage):
    #not done yet
    def __init__(self, d_in, k=5, distance="euclidean", sparse=True):
        super(dDGM, self).__init__()
        self.d_in = d_in
        self.k = k

        self.temperature = nn.Parameter(torch.tensor(1.).float())

    def forward(self, X, A_0= None, fix_DGM=False):
        pass

