import torch
import argparse
import skip_models
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
import torch_geometric.nn as pyg_nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch_geometric.datasets import TUDataset
from sklearn.metrics import *
from torch.nn import Sequential, Linear, ReLU
from deepsnap.dataset import GraphDataset
from deepsnap.batch import Batch
from transforms import *

# torch.manual_seed(0)
# np.random.seed(0)

def arg_parse():
    parser = argparse.ArgumentParser(description='Graph classification arguments.')

    parser.add_argument('--device', type=str,
                        help='CPU / GPU device.')
    parser.add_argument('--epochs', type=int,
                        help='Number of epochs to train.')
    parser.add_argument('--dataset', type=str,
                        help='Graph classification dataset. dd or enzymes.')
    parser.add_argument('--model', type=str,
                        help='GCN, GAT, GraphSAGE or GIN.')
    parser.add_argument('--batch_size', type=int,
                        help='Batch size for graph classification.')
    parser.add_argument('--hidden_dim', type=int,
                        help='Hidden dimension of GNN.')
    parser.add_argument('--num_layers', type=int,
                        help='Number of graph convolution layers.')
    parser.add_argument('--opt', type=str,
                        help='Optimizer such as adam, sgd, rmsprop or adagrad.')
    parser.add_argument('--weight_decay', type=float,
                        help='Weight decay.')
    parser.add_argument('--dropout', type=float,
                        help='The dropout ratio.')
    parser.add_argument('--lr', type=float,
                        help='Learning rate.')
    parser.add_argument('--skip', type=str,
                        help='Skip connections for GCN, GAT or GraphSAGE if specified as last.')
    parser.add_argument('--transform_dataset', type=str,
                        help='apply transform to the whole dataset.')
    parser.add_argument('--transform_batch', type=str,
                        help='apply transform to each batch.')
    parser.add_argument('--radius', type=int,
                        help='Radius of mini-batch ego networks')

    parser.set_defaults(
            device='cuda:0', 
            epochs=500,
            dataset='enzymes',
            model='GIN',
            batch_size=32,
            hidden_dim=20,
            num_layers=3,
            opt='adam',
            weight_decay=5e-4,
            dropout=0.2,
            lr=0.001,
            skip=None,
            transform_dataset=None,
            transform_batch=None,
            radius=3
    )
    return parser.parse_args()


def get_transform(name):
    transform_funcs = {
        'ego': ego_nets,
        'path': path_len,
    }
    assert name in transform_funcs.keys(), \
        'Transform function \'{}\' not supported'.format(name)
    return transform_funcs[name]


def build_optimizer(args, params):
    weight_decay = args.weight_decay
    filter_fn = filter(lambda p : p.requires_grad, params)
    if args.opt == 'adam':
        optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter_fn, lr=args.lr, momentum=0.95, weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'adagrad':
        optimizer = optim.Adagrad(filter_fn, lr=args.lr, weight_decay=weight_decay)
    return optimizer

class GIN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, args):
        super(GIN, self).__init__()
        self.num_layers = args.num_layers
        
        self.pre_mp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim))

        self.convs = nn.ModuleList()
        self.nn1 = Sequential(Linear(hidden_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
        self.convs.append(pyg_nn.GINConv(self.nn1))
        for l in range(args.num_layers-1):
            self.nnk = Sequential(Linear(hidden_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
            self.convs.append(pyg_nn.GINConv(self.nnk))

        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), ReLU(), nn.Linear(hidden_dim, output_dim))

    def forward(self, data):
        x, edge_index, batch = data.node_feature, data.edge_index, data.batch
        x = self.pre_mp(x)
        for i in range(len(self.convs)):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
        x = pyg_nn.global_add_pool(x, batch)
        x = self.post_mp(x)
        x = F.log_softmax(x, dim=1)
        return x

    def loss(self, pred, label):
        return F.nll_loss(pred, label)

class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, args):
        super(GNN, self).__init__()
        self.dropout = args.dropout
        self.num_layers = args.num_layers

        conv_model = self.build_conv_model(args.model)
        self.convs = nn.ModuleList()
        self.convs.append(conv_model(input_dim, hidden_dim))

        for l in range(args.num_layers-1):
            self.convs.append(conv_model(hidden_dim, hidden_dim))

        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Dropout(args.dropout), 
            ReLU(),
            nn.Linear(hidden_dim, output_dim))

    def build_conv_model(self, model_type):
        if model_type == 'GCN':
            return pyg_nn.GCNConv
        elif model_type == 'GAT':
       	    return pyg_nn.GATConv
        elif model_type == "GraphSage":
            return pyg_nn.SAGEConv
        else:
            raise ValueError("Model {} unavailable, please add it to GNN.build_conv_model.".format(model_type))

    def forward(self, data):
        x, edge_index, batch = data.node_feature, data.edge_index, data.batch

        for i in range(len(self.convs)):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # x = pyg_nn.global_mean_pool(x, batch)
        x = pyg_nn.global_add_pool(x, batch)
        x = self.post_mp(x)
        x = F.log_softmax(x, dim=1)
        return x

    def loss(self, pred, label):
        return F.nll_loss(pred, label)

def train(train_loader, val_loader, test_loader, args, num_node_features, num_classes, device="cpu"):
    if args.skip is not None:
        model_cls = skip_models.SkipLastGNN
    elif args.model == "GIN":
        model_cls = GIN
    else:
        model_cls = GNN

    model = model_cls(num_node_features, args.hidden_dim, num_classes, args).to(device)
    opt = build_optimizer(args, model.parameters())

    for epoch in range(args.epochs):
        total_loss = 0
        model.train()
        num_graphs = 0
        for batch in train_loader:
            if args.transform_batch is not None:
                trans_func = get_transform(args.transform_batch)
                batch.apply_transform(trans_func, radius=args.radius)
            batch.to(device)
            opt.zero_grad()
            pred = model(batch)
            label = batch.graph_label
            loss = model.loss(pred, label)
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch.num_graphs
            num_graphs += batch.num_graphs
        total_loss /= num_graphs

        train_acc = test(train_loader, model, args, device)
        val_acc = test(val_loader, model, args, device)
        test_acc = test(test_loader, model, args, device)
        print("Epoch {}: Train: {:.4f}, Validation: {:.4f}. Test: {:.4f}, Loss: {:.4f}".format(epoch + 1, train_acc, val_acc, test_acc, total_loss))

def test(loader, model, args, device='cuda'):
    model.eval()

    correct = 0
    num_graphs = 0
    for batch in loader:
        if args.transform_batch is not None:
            trans_func = get_transform(args.transform_batch)
            batch.apply_transform(trans_func, radius=args.radius)
        batch.to(device)
        with torch.no_grad():
            pred = model(batch).max(dim=1)[1]
            label = batch.graph_label
        correct += pred.eq(label).sum().item()
        num_graphs += batch.num_graphs
    # print("loader len {}".format(num_graphs))
    return correct / num_graphs

if __name__ == "__main__":
    args = arg_parse()

    if args.dataset == 'enzymes':
        pyg_dataset = TUDataset('./enzymes', 'ENZYMES')
    elif args.dataset == 'dd':
        pyg_dataset = TUDataset('./dd', 'DD')
    else:
        raise ValueError("Unsupported dataset.")

    graphs = GraphDataset.pyg_to_graphs(pyg_dataset)

    dataset = GraphDataset(graphs, task="graph")
    datasets = {}
    datasets['train'], datasets['val'], datasets['test'] = dataset.split(
            transductive=False, split_ratio = [0.8, 0.1, 0.1])

    if args.transform_dataset is not None:
        trans_func = get_transform(args.transform_dataset)
        for _, dataset in datasets.items():
            dataset.apply_transform(trans_func, radius=args.radius)

    dataloaders = {split: DataLoader(
                dataset, collate_fn=Batch.collate(), 
                batch_size=args.batch_size, shuffle=True)
                for split, dataset in datasets.items()}

    num_classes = datasets['train'].num_graph_labels
    num_node_features = datasets['train'].num_node_features

    train(dataloaders['train'], dataloaders['val'], dataloaders['test'], 
            args, num_node_features, num_classes, args.device)
