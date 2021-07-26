import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch_scatter import scatter_mean
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj, Size, Tensor

from neuralnet.egnn_sparse import *


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
        

class H5Net(nn.Module):
    def __init__(
        self,
        embedding_dim=24,
        n_kernels=2,
        n_mlp=3,
        mlp_dim=256,
        n_outputs=4,
        m_dim=64,
        initialize_weights=True,
        fourier_features=4,
        aggr="sum",
    ):
        super(H5Net, self).__init__()

        self.pos_dim = 3
        self.m_dim = m_dim
        self.embedding_dim = embedding_dim
        self.n_kernels = n_kernels
        self.n_mlp = n_mlp
        self.mlp_dim = mlp_dim
        self.n_outputs = n_outputs
        self.initialize_weights = initialize_weights
        self.fourier_features = fourier_features
        self.aggr = aggr

        # Atom type Embedding
        self.embedding = nn.Embedding(
            num_embeddings=22, embedding_dim=self.embedding_dim
        )

        # Previous ffnn
        self.emb_ffl = nn.Linear(self.embedding_dim, self.embedding_dim)

        # Kernel
        self.kernel_dim = self.embedding_dim
        self.kernels = nn.ModuleList()
        for _ in range(self.n_kernels):
            self.kernels.append(
                EGNN_sparse(
                    feats_dim=self.kernel_dim,
                    pos_dim=self.pos_dim,
                    m_dim=self.m_dim,
                    fourier_features=self.fourier_features,
                    aggr=self.aggr,
                )
            )

        # MLP 1
        self.ffnn = nn.ModuleList()
        input_ffnn = self.kernel_dim * (self.n_kernels + 1)
        self.ffnn.append(nn.Linear(input_ffnn, mlp_dim))
        for _ in range(self.n_mlp - 1):
            self.ffnn.append(nn.Linear(self.mlp_dim, self.mlp_dim))

        self.ffnn.append(nn.Linear(self.mlp_dim, self.n_outputs))

        # Initialize weights
        if self.initialize_weights:
            self.kernels.apply(weights_init)
            self.ffnn.apply(weights_init)
            nn.init.xavier_uniform_(self.embedding.weight)
         
         # Write attention on graph to use weighted feature aggregation

        
    def forward(self, batch):
        features = self.embedding(batch.z)
        features = self.emb_ffl(features)
        features = torch.cat([batch.pos, features], dim=1)  # ===> this breaks the rotation equivariance! use  distance as feature? ==> we use pos_dim: to ignore the location within the kernels,  but use them in the ffl (the mean values I guess.. weird!)
        feature_list = []
        feature_list.append(features[:, self.pos_dim:])

        for kernel in self.kernels:
            features = kernel(
                x=features,
                edge_index=batch.edge_index,
            )

            feature_list.append(features[:, self.pos_dim:])

        features = F.silu(torch.cat(feature_list, dim=1))

        for mlp in self.ffnn[:-1]:
            features = F.silu(mlp(features))

        features = scatter_mean(features, batch.batch, dim=0)  # why scatter mean?

        features = torch.sigmoid(self.ffnn[-1](features))

        return features

    

