import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils


class SkipLastGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, args):
        super(SkipLastGNN, self).__init__()
        self.args = args
        self.dropout = args.dropout
        self.num_layers = args.num_layers

        self.pre_mp = nn.Sequential(nn.Linear(input_dim, hidden_dim))

        conv_model = self.build_conv_model(args.model)
        self.convs = nn.ModuleList()

        if args.skip == 'learnable':
            self.learnable_skip = nn.Parameter(torch.ones(self.num_layers, self.num_layers))

        for l in range(args.num_layers):
            if args.skip == 'all' or 'learnable':
                hidden_input_dim = hidden_dim * (l + 1)
            elif args.skip == 'last':
                hidden_input_dim = hidden_dim
            else:
                raise ValueError(f'Unknown skip option {args.skip}')
            self.convs.append(conv_model(hidden_input_dim, hidden_dim))

        post_input_dim = hidden_dim * (args.num_layers + 1)
        self.post_mp = nn.Sequential(
            nn.Linear(post_input_dim, hidden_dim), nn.Dropout(args.dropout), nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, output_dim))

    def build_conv_model(self, model_type):
        if model_type == 'GCN':
            return pyg_nn.GCNConv
        elif model_type == 'GAT':
       	    return pyg_nn.GATConv
        elif model_type == "GraphSage":
            return pyg_nn.SAGEConv
        elif model_type == "Graph":
            return pyg_nn.GraphConv
        elif model_type == "Simple":
            if self.args.skip == 'all':
                raise ValueError("SimpleConv does not have parameter and does not support full skip connections")
            return SimpleConv
        else:
            raise ValueError("Model_type {} unavailable, please add it to GNN.build_conv_model.".format(model_type))

    def forward(self, data):
        x, edge_index, batch = data.node_feature, data.edge_index, data.batch
        x = self.pre_mp(x)
        num_nodes = x.size(0)

        # [num nodes x current num layer x hidden_dim]
        all_emb = x.unsqueeze(1)
        # [num nodes x (curr num layer * hidden_dim)]
        emb = x
        for i in range(len(self.convs)):
            if self.args.skip == 'learnable':
                skip_vals = self.learnable_skip[i, :i+1].unsqueeze(0).unsqueeze(-1)
                curr_emb = all_emb * torch.sigmoid(skip_vals)
                curr_emb = curr_emb.view(num_nodes, -1)
                x = self.convs[i](curr_emb, edge_index)
            if self.args.skip == 'all' or self.args.skip == 'learnable':
                x = self.convs[i](emb, edge_index)
            else:
                x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            emb = torch.cat((emb, x), 1)
            if self.args.skip == 'learnable':
                all_emb = torch.cat((all_emb, x.unsqueeze(1)), 1)

        # x = pyg_nn.global_mean_pool(x, batch)
        emb = pyg_nn.global_add_pool(emb, batch)
        emb = self.post_mp(emb)
        out = F.log_softmax(emb, dim=1)
        return out

    def loss(self, pred, label):
        return F.nll_loss(pred, label)


class SimpleConv(pyg_nn.MessagePassing):
    def __init__(self, in_channels, out_channels, cache=False):
        super(SimpleConv, self).__init__(aggr='add')
        self.cache = cache
        self.reset_cache()

    def reset_cache(self):
        self.cached_norm = None
        
    def forward(self, x, edge_index, add_self=False):
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        emb = self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)
        if add_self:
            emb = emb + x
        return emb

    def message(self, x_i, x_j, edge_index, size):
        if self.cached_norm is None:
            row, col = edge_index
            deg = pyg_utils.degree(row, size[0], dtype=x_j.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            if self.cache:
                self.cached_norm = norm
        else:
            norm = self.cached_norm
        return norm.view(-1, 1) * x_j

