from functools import reduce
import random

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils

import utils

# GNN -> concat -> graph classification baseline
class BaselineMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, args):
        super(BaselineMLP, self).__init__()
        self.emb_model = GNNStack(input_dim, hidden_dim, hidden_dim, args)
        self.mlp = nn.Sequential(nn.Linear(2 * hidden_dim, 2))

    def forward(self, emb_motif, emb_motif_mod):
        pred = self.mlp(torch.cat((emb_motif, emb_motif_mod), dim=1))
        pred = F.log_softmax(pred, dim=1)
        return pred

    def predict(self, pred):
        return pred.argmax(dim=1)

    def criterion(self, pred, label):
        return F.nll_loss(pred, label)

# Graph embedding model
class GNNStack(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, args, task='node'):
        super(GNNStack, self).__init__()
        self.input_dim = input_dim
        conv_model = self.build_conv_model(args.conv_type)
        self.convs = nn.ModuleList()
        self.convs.append(conv_model(input_dim, hidden_dim))
        self.conv_type = args.conv_type
        assert (args.n_layers >= 1), 'Number of layers is not >=1'
        for l in range(args.n_layers-1):
            self.convs.append(conv_model(hidden_dim, hidden_dim))

        if self.conv_type == "gated":
            self.glob_soft_attn = pyg_nn.GlobalAttention(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, 1)))

        # post-message-passing
        self.post_mp = nn.Sequential(
            nn.Linear(args.n_layers*hidden_dim, hidden_dim),
            nn.Dropout(args.dropout), 
            nn.Linear(hidden_dim, output_dim))

        self.task = task
        if not (self.task == 'node' or self.task == 'graph'):
            raise RuntimeError('Unknown task.')

        self.dropout = args.dropout
        self.num_layers = args.n_layers
        self.hidden_dim = hidden_dim

    def build_conv_model(self, model_type):
        if model_type == "GCN":
            return pyg_nn.GCNConv
        elif model_type == "GIN":
            return lambda i, h: GINConv(nn.Sequential(
                nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h)))
        elif model_type == "SAGE":
            return SAGEConv
        elif model_type == "graph":
            return pyg_nn.GraphConv
        elif model_type == "GAT":
            return pyg_nn.GATConv
        else:
            print("unrecognized model type")

    def forward(self, data):
        x, edge_index, batch = data.node_feature, data.edge_index, data.batch
        #print(batch)

        layers_out = []
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            layers_out.append(pyg_nn.global_add_pool(x, batch))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.relu(x)

        layers_out = torch.cat(layers_out, dim=-1)

        x = self.post_mp(layers_out)

        return x

class SAGEConv(pyg_nn.MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(SAGEConv, self).__init__(aggr='add')

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin = nn.Linear(in_channels, out_channels)
        self.lin_update = nn.Linear(out_channels + in_channels,
            out_channels)

    def forward(self, x, edge_index, edge_weight=None, size=None,
                res_n_id=None):
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)

        return self.propagate(edge_index, size=size, x=x,
                              edge_weight=edge_weight, res_n_id=res_n_id)

    def message(self, x_j, edge_weight):
        return self.lin(x_j)

    def update(self, aggr_out, x, res_n_id):
        aggr_out = torch.cat([aggr_out, x], dim=-1)

        aggr_out = self.lin_update(aggr_out)
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
