import torch
from torch.utils.data import Dataset
from torch_geometric.datasets import Planetoid
import numpy as np
from einops import rearrange
from typing import Tuple
import random


class GraphDataset(Dataset):
    def __init__(self, graph_name: str, split: str, ratio: float=0.8, norm_feature: bool=False, norm_adj: bool=False):
        if graph_name == "Cora":
            data = Planetoid(root='./data', name='Cora')[0]
            x_all = data.x
            if norm_feature == True:
                pass # TODO
            y_all = data.y
            self.edge_index = data.edge_index
            if norm_feature == True:
                pass # TODO: edge-index, not adjacency matrix
            self.N = x_all.shape[0]
            self.d = x_all.shape[1]
            self.c = int(y_all.max() + 1)
            # self.train_ids = list(random.sample(range(self.N), int(ratio*self.N)))
            # self.val_ids = list(set(range(range(self.N))).difference(set(self.train_ids)))
            
            self.split = split

            if split == 'train':
                self.node_ids = torch.LongTensor(range(int(ratio*self.N)))
            elif split == 'val':
                self.node_ids = torch.LongTensor(range(int(ratio*self.N), self.N))
            elif split == 'all':
                self.node_ids = torch.LongTensor(range(self.N))

            self.x = x_all[self.node_ids]
            self.y = y_all[self.node_ids]


    def __len__(self):
        if self.split == 'train':
            return len(self.train_ids)
        elif self.split == "val":
            return len(self.val_ids)

    def __getitem__(self, idx: int):
        # idx: [0, __len__()]
        return {'id': idx, "x": self.x[idx], "y": self.y[idx]} 


if __name__ == "__main__":
    data = GraphDataset("cora", "train")
