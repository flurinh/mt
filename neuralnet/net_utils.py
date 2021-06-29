import networkx as nx
import numpy as np
import torch, h5py
from torch_geometric.data import Data, Dataset
from torch_geometric.utils.undirected import to_undirected
from torch_geometric.utils import add_self_loops
import pickle


class DatasetSingletask(Dataset):
    def __init__(
        self,
        h5file="qt_dict.h5",
    ):
        # Read h5: PDB -> props
        self.h5f = h5py.File(h5file, "r")
        print("len(self.h5f) Data: ", len(self.h5f))

        # Make dict on the fly: idx -> pdb 
        with open(h5file[:-4]+".txt", 'r') as f:
            lines = f.readlines()

        pdbs = [x.split("/")[-2] for x in lines]
        nums = list(range(0, len(pdbs)))

        self.idx2pdb = {}

        for x in range(len(pdbs)):
            dict={nums[x]:pdbs[x]}
            self.idx2pdb.update(dict)

    def __getitem__(self, idx):
        
        # get pdb from id 
        pdb = self.idx2pdb[idx]
            
        # nodes coordinates and target
        identity = torch.LongTensor(self.h5f[str(pdb)]["idx"])
        atomids = torch.LongTensor(self.h5f[str(pdb)]["atomids"])
        target = torch.FloatTensor(self.h5f[str(pdb)]["log_affinity"])
        coords = torch.FloatTensor(self.h5f[str(pdb)]["coords"])
        edge_index = torch.LongTensor(self.h5f[str(pdb)]["edge_2d"])

        # edges
        # edge_index = np.array(nx.complete_graph(atomids.size(0)).edges())
        # edge_index = to_undirected(torch.from_numpy(edge_index).t().contiguous())
        # edge_index, _ = add_self_loops(edge_index, num_nodes=coords.shape[0])

        # graph object
        graph_data = Data(
            atomids=atomids,
            identity=identity,
            coords=coords,
            edge_index=edge_index,
            target=target,
            num_nodes=atomids.size(0),
        )

        return graph_data

    def __len__(self):
        return len(self.h5f)


if __name__ == '__main__':
    from torch_geometric.data import DataLoader

    h5file = "data/train_paths100_.h5"

    h5f = h5py.File(h5file, "r")
    print("len(self.h5f) Data: ", len(h5f))
    train_data = DatasetSingletask(h5file=h5file)
    train_pbar = DataLoader(train_data, batch_size=2, shuffle=False, num_workers=4)
    print("Training set: ", len(train_data))

    for g_batch in train_pbar:
        print(g_batch.target.size())
        print(g_batch.edge_index.size())
