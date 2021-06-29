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
        

class DeltaNetMolecular(nn.Module):
    def __init__(
        self,
        embedding_dim=128,
        n_kernels=2,
        n_mlp=3,
        mlp_dim=256,
        n_outputs=1,
        m_dim=64,
        initialize_weights=True,
        fourier_features=4,
        aggr="mean",
    ):
        super(DeltaNetMolecular, self).__init__()

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
            num_embeddings=11, embedding_dim=self.embedding_dim
        )

        # Ligands/Protein type embedding Embedding
        self.embedding_dim_id = int(self.embedding_dim / 2)
        self.embedding_id = nn.Embedding(
            num_embeddings=2, embedding_dim=self.embedding_dim_id
        )

        # Previous fnn
        self.initialfnn = nn.Linear(self.embedding_dim_id + self.embedding_dim, self.embedding_dim)

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
        self.fnn = nn.ModuleList()
        input_fnn = self.kernel_dim * (self.n_kernels + 1)
        self.fnn.append(nn.Linear(input_fnn, mlp_dim))
        for _ in range(self.n_mlp - 1):
            self.fnn.append(nn.Linear(self.mlp_dim, self.mlp_dim))

        # MLP 2
        self.fnn2 = nn.ModuleList()
        for _ in range(self.n_mlp - 1):
            self.fnn2.append(nn.Linear(self.mlp_dim, self.mlp_dim))
        self.fnn2.append(nn.Linear(self.mlp_dim, self.n_outputs))

        # Initialize weights
        if self.initialize_weights:
            self.kernels.apply(weights_init)
            self.fnn.apply(weights_init)
            self.fnn2.apply(weights_init)
            nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, g_batch):

        features = self.embedding(g_batch.atomids)
        featuresid = self.embedding_id(g_batch.identity)
        features = F.silu(self.initialfnn(torch.cat([featuresid, features], dim=1)))
        features = torch.cat([g_batch.coords, features], dim=1)

        feature_list = []
        feature_list.append(features[:, self.pos_dim :])

        for kernel in self.kernels:
            features = kernel(
                x=features,
                edge_index=g_batch.edge_index,
            )

            feature_list.append(features[:, self.pos_dim :])

        features = F.silu(torch.cat(feature_list, dim=1))

        for mlp in self.fnn:
            features = F.silu(mlp(features))

        features = scatter_mean(features, g_batch.batch, dim=0)

        for mlp in self.fnn2[:-1]:
            features = F.silu(mlp(features))

        features = self.fnn2[-1](features)

        return features

    

