import torch
import torch.nn as nn
import torch_geometric

class base_edage:
    def __init__(self):
        super().__init__()
        self.pdist = nn.PairwiseDistance(p=2)
    def forward(self, x, A_0=None):
        pass
    def reset_parameters(self):
        pass

    def distance_calculation(self, X_hat):
        return torch.norm(X_hat[:, None] - X_hat, dim=2, p=2)
        #return self.pdist(X_hat, X_hat)


class cDGM(nn.Module):

    def __init__(self, d_in: int, d_out: int = None):
        super().__init__()

        # define layers for f_{\Theta}
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

