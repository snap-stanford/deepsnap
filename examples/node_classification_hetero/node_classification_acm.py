import copy
import torch
import argparse
import deepsnap
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn

from sklearn.metrics import f1_score
from deepsnap.hetero_gnn import forward_op, HeteroSAGEConv
from deepsnap.hetero_graph import HeteroGraph
from torch_sparse import SparseTensor, matmul

def arg_parse():
    parser = argparse.ArgumentParser(description='ACM.')

    parser.add_argument('--device', type=str,
                        help='CPU / GPU device.')
    parser.add_argument('--epochs', type=int,
                        help='Number of epochs to train.')
    parser.add_argument('--hidden_size', type=int,
                        help='GNN layer hidden dimension size.')
    parser.add_argument('--lr', type=float,
                        help='Learning rate.')
    parser.add_argument('--eps', type=float,
                        help='Batchnorm eps.')
    parser.add_argument('--attn_size', type=int,
                        help='Attention size.')
    parser.add_argument('--weight_decay', type=int,
                        help='Weight decay.')

    parser.set_defaults(
        device='cuda:0',
        epochs=100,
        hidden_size=64,
        lr=0.003,
        eps=1.0,
        attn_size=32,
        weight_decay=1e-5,
    )
    return parser.parse_args()

class HeteroGNNWrapperConv(deepsnap.hetero_gnn.HeteroConv):
    def __init__(self, convs, args, aggr="mean"):
        super(HeteroGNNWrapperConv, self).__init__(convs, None)
        self.aggr = aggr

        # Map the index and message type
        self.mapping = {}

        # A numpy array that stores the final attention probability
        self.alpha = None

        self.attn_proj = None

        if self.aggr == "attn":
            self.attn_proj = nn.Sequential(
                nn.Linear(args.hidden_size, args.attn_size),
                nn.Tanh(),
                nn.Linear(args.attn_size, 1, bias=False)
            )
    
    def reset_parameters(self):
        super(HeteroConvWrapper, self).reset_parameters()
        if self.aggr == "attn":
            for layer in self.attn_proj.children():
                layer.reset_parameters()
    
    def forward(self, node_features, edge_indices):
        message_type_emb = {}
        for message_key, message_type in edge_indices.items():
            src_type, edge_type, dst_type = message_key
            node_feature_src = node_features[src_type]
            node_feature_dst = node_features[dst_type]
            edge_index = edge_indices[message_key]
            message_type_emb[message_key] = (
                self.convs[message_key](
                    node_feature_src,
                    node_feature_dst,
                    edge_index,
                )
            )
        node_emb = {dst: [] for _, _, dst in message_type_emb.keys()}
        mapping = {}        
        for (src, edge_type, dst), item in message_type_emb.items():
            mapping[len(node_emb[dst])] = (src, edge_type, dst)
            node_emb[dst].append(item)
        self.mapping = mapping
        for node_type, embs in node_emb.items():
            if len(embs) == 1:
                node_emb[node_type] = embs[0]
            else:
                node_emb[node_type] = self.aggregate(embs)
        return node_emb
    
    def aggregate(self, xs):
        if self.aggr == "mean":
            x = torch.stack(xs, dim=-1)
            return x.mean(dim=-1)

        elif self.aggr == "attn":
            N = xs[0].shape[0] # Number of nodes for that node type
            M = len(xs) # Number of message types for that node type

            x = torch.cat(xs, dim=0).view(M, N, -1) # M * N * D
            z = self.attn_proj(x).view(M, N) # M * N * 1
            z = z.mean(1) # M * 1
            alpha = torch.softmax(z, dim=0) # M * 1

            # Store the attention result to self.alpha as np array
            self.alpha = alpha.view(-1).data.cpu().numpy()
  
            alpha = alpha.view(M, 1, 1)
            x = x * alpha
            return x.sum(dim=0)

def generate_convs(hetero_graph, conv, hidden_size, first_layer=False):
    convs = {}
    for message_type in hetero_graph.message_types:
        if first_layer is True:
            src_type = message_type[0]
            dst_type = message_type[2]
            src_size = hetero_graph.num_node_features(src_type)
            dst_size = hetero_graph.num_node_features(dst_type)
            convs[message_type] = conv(src_size, hidden_size, dst_size, remove_self_loop=False)
        else:
            convs[message_type] = conv(hidden_size, hidden_size, hidden_size, remove_self_loop=False)    
    return convs

class HeteroGNN(torch.nn.Module):
    def __init__(self, hetero_graph, args, aggr="mean"):
        super(HeteroGNN, self).__init__()

        self.aggr = aggr
        self.hidden_size = args.hidden_size

        self.convs1 = None
        self.convs2 = None

        self.bns1 = nn.ModuleDict()
        self.bns2 = nn.ModuleDict()
        self.relus1 = nn.ModuleDict()
        self.relus2 = nn.ModuleDict()
        self.post_mps = nn.ModuleDict()

        convs1 = generate_convs(hetero_graph, HeteroSAGEConv, self.hidden_size, first_layer=True)
        convs2 = generate_convs(hetero_graph, HeteroSAGEConv, self.hidden_size)

        self.convs1 = HeteroGNNWrapperConv(convs1, args, aggr=self.aggr)
        self.convs2 = HeteroGNNWrapperConv(convs2, args, aggr=self.aggr)

        for node_type in hetero_graph.node_types:
            self.bns1[node_type] = torch.nn.BatchNorm1d(self.hidden_size, eps=args.eps)
            self.bns2[node_type] = torch.nn.BatchNorm1d(self.hidden_size, eps=args.eps)
            self.post_mps[node_type] = nn.Linear(self.hidden_size, hetero_graph.num_node_labels(node_type))
            self.relus1[node_type] = nn.LeakyReLU()
            self.relus2[node_type] = nn.LeakyReLU()

    def forward(self, node_feature, edge_index):
        x = node_feature
        x = self.convs1(x, edge_index)
        x = forward_op(x, self.bns1)
        x = forward_op(x, self.relus1)
        x = self.convs2(x, edge_index)
        x = forward_op(x, self.bns2)
        x = forward_op(x, self.relus2)
        x = forward_op(x, self.post_mps)
        return x

    def loss(self, preds, y, indices):
        loss = 0
        loss_func = F.cross_entropy
        for node_type in preds:
            idx = indices[node_type]
            loss += loss_func(preds[node_type][idx], y[node_type][idx])
        return loss

def train(model, optimizer, hetero_graph, train_idx):
    model.train()
    optimizer.zero_grad()
    preds = model(hetero_graph.node_feature, hetero_graph.edge_index)
    loss = model.loss(preds, hetero_graph.node_label, train_idx)
    loss.backward()
    optimizer.step()
    return loss.item()

def test(model, graph, indices, best_model=None, best_val=0):
    model.eval()
    accs = []
    for index in indices:
        preds = model(graph.node_feature, graph.edge_index)
        num_node_types = 0
        micro = 0
        macro = 0
        for node_type in preds:
            idx = index[node_type]
            pred = preds[node_type][idx]
            pred = pred.max(1)[1]
            label_np = graph.node_label[node_type][idx].cpu().numpy()
            pred_np = pred.cpu().numpy()
            micro = f1_score(label_np, pred_np, average='micro')
            macro = f1_score(label_np, pred_np, average='macro')
            num_node_types += 1
        # Averaging f1 score might not make sense, but in our example we only
        # have one node type
        micro /= num_node_types
        macro /= num_node_types
        accs.append((micro, macro))
    if accs[1][0] > best_val:
        best_val = accs[1][0]
        best_model = copy.deepcopy(model)
    return accs, best_model, best_val

if __name__ == '__main__':
    args = arg_parse()

    print("Device: {}".format(args.device))

    # Load the data
    data = torch.load("data/acm.pkl")

    # Message types
    message_type_1 = ("paper", "author", "paper")
    message_type_2 = ("paper", "subject", "paper")

    # Dictionary of edge indices
    edge_index = {}
    edge_index[message_type_1] = data['pap']
    edge_index[message_type_2] = data['psp']

    # Dictionary of node features
    node_feature = {}
    node_feature["paper"] = data['feature']

    # Dictionary of node labels
    node_label = {}
    node_label["paper"] = data['label']

    # Load the train, validation and test indices
    train_idx = {"paper": data['train_idx'].to(args.device)}
    val_idx = {"paper": data['val_idx'].to(args.device)}
    test_idx = {"paper": data['test_idx'].to(args.device)}

    # Construct a deepsnap tensor backend HeteroGraph
    hetero_graph = HeteroGraph(
        node_feature=node_feature,
        node_label=node_label,
        edge_index=edge_index,
        directed=True
    )

    print(f"ACM heterogeneous graph: {hetero_graph.num_nodes()} nodes, {hetero_graph.num_edges()} edges")

    # Node feature and node label to device
    for key in hetero_graph.node_feature:
        hetero_graph.node_feature[key] = hetero_graph.node_feature[key].to(args.device)
    for key in hetero_graph.node_label:
        hetero_graph.node_label[key] = hetero_graph.node_label[key].to(args.device)

    # Edge_index to sparse tensor and to device
    for key in hetero_graph.edge_index:
        edge_index = hetero_graph.edge_index[key]
        adj = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(hetero_graph.num_nodes('paper'), hetero_graph.num_nodes('paper')))
        hetero_graph.edge_index[key] = adj.t().to(args.device)
    print(hetero_graph.edge_index[message_type_1])
    print(hetero_graph.edge_index[message_type_2])


    # Mean aggregation
    best_model = None
    best_val = 0

    model = HeteroGNN(hetero_graph, args, aggr="mean").to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        loss = train(model, optimizer, hetero_graph, train_idx)
        accs, best_model, best_val = test(model, hetero_graph, [train_idx, val_idx, test_idx], best_model, best_val)
        print(
            f"Epoch {epoch + 1}: loss {round(loss, 5)}, "
            f"train micro {round(accs[0][0] * 100, 2)}%, train macro {round(accs[0][1] * 100, 2)}%, "
            f"valid micro {round(accs[1][0] * 100, 2)}%, valid macro {round(accs[1][1] * 100, 2)}%, "
            f"test micro {round(accs[2][0] * 100, 2)}%, test macro {round(accs[2][1] * 100, 2)}%"
        )
    best_accs, _, _ = test(best_model, hetero_graph, [train_idx, val_idx, test_idx])
    print(
        f"Best model: "
        f"train micro {round(best_accs[0][0] * 100, 2)}%, train macro {round(best_accs[0][1] * 100, 2)}%, "
        f"valid micro {round(best_accs[1][0] * 100, 2)}%, valid macro {round(best_accs[1][1] * 100, 2)}%, "
        f"test micro {round(best_accs[2][0] * 100, 2)}%, test macro {round(best_accs[2][1] * 100, 2)}%"
    )


    # Attention aggregation
    best_model = None
    best_val = 0

    output_size = hetero_graph.num_node_labels('paper')
    model = HeteroGNN(hetero_graph, args, aggr="attn").to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        loss = train(model, optimizer, hetero_graph, train_idx)
        accs, best_model, best_val = test(model, hetero_graph, [train_idx, val_idx, test_idx], best_model, best_val)
        print(
            f"Epoch {epoch + 1}: loss {round(loss, 5)}, "
            f"train micro {round(accs[0][0] * 100, 2)}%, train macro {round(accs[0][1] * 100, 2)}%, "
            f"valid micro {round(accs[1][0] * 100, 2)}%, valid macro {round(accs[1][1] * 100, 2)}%, "
            f"test micro {round(accs[2][0] * 100, 2)}%, test macro {round(accs[2][1] * 100, 2)}%"
        )
    best_accs, _, _ = test(best_model, hetero_graph, [train_idx, val_idx, test_idx])
    print(
        f"Best model: "
        f"train micro {round(best_accs[0][0] * 100, 2)}%, train macro {round(best_accs[0][1] * 100, 2)}%, "
        f"valid micro {round(best_accs[1][0] * 100, 2)}%, valid macro {round(best_accs[1][1] * 100, 2)}%, "
        f"test micro {round(best_accs[2][0] * 100, 2)}%, test macro {round(best_accs[2][1] * 100, 2)}%"
    )

    if model.convs1.alpha is not None and model.convs2.alpha is not None:
        for idx, message_type in model.convs1.mapping.items():
            print(f"Layer 1 has attention {model.convs1.alpha[idx]} on message type {message_type}")
        for idx, message_type in model.convs2.mapping.items():
            print(f"Layer 2 has attention {model.convs2.alpha[idx]} on message type {message_type}")