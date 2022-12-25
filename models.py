import torch
from torch import nn
import numpy as np
from einops import rearrange
from torch_geometric.nn import MessagePassing, GATConv, GCNConv, SAGEConv
import torch.nn.functional as F

class GCN(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim_ls: list, output_dim: int):
        super().__init__()
        layers = []
        self.conv0 = GCNConv(input_dim, hidden_dim_ls[0])
        layers.append(self.conv0)
        layers.append(nn.ReLU())
        self.num_hidden = len(hidden_dim_ls)
        for i in range(self.num_hidden-1):
            setattr(self, f"conv{i+1}", GCNConv(hidden_dim_ls[i], hidden_dim_ls[i+1]))
            layers.append(getattr(self, f"conv{i+1}"))
            layers.append(nn.ReLU())
        setattr(self, f"conv{self.num_hidden}", GCNConv(hidden_dim_ls[-1], output_dim))
        layers.append(getattr(self, f"conv{self.num_hidden}"))
        self.layers = nn.Sequential(*layers)

    def forward(self, x, edge_index):
        # x = self.layers(x, edge_index)
        x = self.conv0(x, edge_index)
        return x
        # return F.log_softmax(x, dim=1)

        # TODO
        # x = F.dropout(x, training=self.training)
        # x = F.normalize(x)

class SAGE(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers,
                 dropout,
                 batchnorm=True):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.batchnorm = batchnorm
        self.num_layers = num_layers
        if self.batchnorm:
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for i in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            if self.batchnorm:
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout
        
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.batchnorm:
            for bn in self.bns:
                bn.reset_parameters()        
        
        
    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers-1:
                if self.batchnorm:
                    x = self.bns[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x       
        # return x.log_softmax(dim=-1)


if __name__ == "__main__":
    gcn = GCN(1433, [256, 16], 7)
    print(gcn)