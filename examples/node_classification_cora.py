import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
import torch_geometric.nn as pyg_nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch_geometric.datasets import Planetoid
from sklearn.metrics import *
from torch.nn import Sequential, Linear, ReLU
from deepsnap.dataset import GraphDataset
from deepsnap.batch import Batch


# torch.manual_seed(0)
# np.random.seed(0)

def arg_parse():
    parser = argparse.ArgumentParser(description='Node classification arguments.')

    parser.add_argument('--device', type=str,
                        help='CPU / GPU device.')
    parser.add_argument('--epochs', type=int,
                        help='Number of epochs to train.')
    parser.add_argument('--dataset', type=str,
                        help='Node classification dataset. Cora, CiteSeer, PubMed')
    parser.add_argument('--model', type=str,
                        help='GCN, GAT, GraphSAGE.')
    parser.add_argument('--batch_size', type=int,
                        help='Batch size for node classification.')
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
    parser.add_argument('--split', type=str,
                        help='Randomly split dataset, or use fixed split in PyG. fixed, random')

    parser.set_defaults(
        device='cuda:0',
        epochs=200,
        dataset='Cora',
        model='GCN',
        batch_size=1,
        hidden_dim=32,
        num_layers=2,
        opt='adam',
        weight_decay=5e-4,
        dropout=0.0,
        lr=0.01,
        split='random'
    )
    return parser.parse_args()


def build_optimizer(args, params):
    weight_decay = args.weight_decay
    filter_fn = filter(lambda p: p.requires_grad, params)
    if args.opt == 'adam':
        optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter_fn, lr=args.lr, momentum=0.95, weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'adagrad':
        optimizer = optim.Adagrad(filter_fn, lr=args.lr, weight_decay=weight_decay)
    return optimizer


class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, args):
        super(GNN, self).__init__()
        self.dropout = args.dropout
        self.num_layers = args.num_layers

        conv_model = self.build_conv_model(args.model)
        self.convs = nn.ModuleList()
        self.convs.append(conv_model(input_dim, hidden_dim))

        for l in range(args.num_layers - 2):
            self.convs.append(conv_model(hidden_dim, hidden_dim))
        self.convs.append(conv_model(hidden_dim, output_dim))

    def build_conv_model(self, model_type):
        if model_type == 'GCN':
            return pyg_nn.GCNConv
        elif model_type == 'GAT':
            return pyg_nn.GATConv
        elif model_type == "GraphSage":
            return pyg_nn.SAGEConv
        else:
            raise ValueError(
                "Model {} unavailable, please add it to GNN.build_conv_model.".format(model_type))

    def forward(self, data):
        x, edge_index, batch = data.node_feature, data.edge_index, data.batch

        for i in range(len(self.convs) - 1):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[len(self.convs) - 1](x, edge_index)
        x = F.log_softmax(x, dim=1)
        return x

    def loss(self, pred, label):
        return F.nll_loss(pred, label)


def train(train_loader, val_loader, test_loader, args, num_node_features, num_classes,
          device="cpu"):
    model_cls = GNN

    model = model_cls(num_node_features, args.hidden_dim, num_classes, args).to(device)
    opt = build_optimizer(args, model.parameters())

    for epoch in range(args.epochs):
        total_loss = 0
        model.train()
        for batch in train_loader:
            batch.to(device)
            opt.zero_grad()
            pred = model(batch)
            label = batch.node_label
            loss = model.loss(pred[batch.node_label_index], label[batch.node_label_index])
            total_loss += loss.item()
            loss.backward()
            opt.step()

        train_acc = test(train_loader, model, device)
        val_acc = test(val_loader, model, device)
        test_acc = test(test_loader, model, device)
        print("Epoch {}: Train: {:.4f}, Validation: {:.4f}. Test: {:.4f}, Loss: {:.4f}".format(
            epoch + 1, train_acc, val_acc, test_acc, total_loss))


def test(loader, model, device='cuda'):
    model.eval()

    for batch in loader:
        batch.to(device)
        logits = model(batch)
        pred = logits[batch.node_label_index].max(1)[1]
        acc = pred.eq(batch.node_label[batch.node_label_index]).sum().item()
        total = batch.node_label_index.shape[0]
        acc /= total
    return acc


if __name__ == "__main__":
    args = arg_parse()
    if args.dataset in ['Cora', 'CiteSeer', 'Pubmed']:
        pyg_dataset = Planetoid('./planetoid', args.dataset,
                                transform=T.TargetIndegree())  # load some format of graph data
    else:
        raise ValueError("Unsupported dataset.")

    if args.split == 'random':
        graphs = GraphDataset.pyg_to_graphs(pyg_dataset, verbose=True,
                                            fixed_split=False)  # transform to our format

        dataset = GraphDataset(graphs, task='node')  # node, edge, link_pred, graph
        dataset_train, dataset_val, dataset_test = dataset.split(
            transductive=True,
            split_ratio=[0.8, 0.1, 0.1])  # transductive split, inductive split

    else:
        graphs_train, graphs_val, graphs_test = \
            GraphDataset.pyg_to_graphs(pyg_dataset, verbose=True,
                                       fixed_split=True)  # transform to our format

        dataset_train, dataset_val, dataset_test = \
            GraphDataset(graphs_train, task='node'), GraphDataset(graphs_val,task='node'), \
            GraphDataset(graphs_test, task='node')

    train_loader = DataLoader(dataset_train, collate_fn=Batch.collate(),
                              batch_size=16)  # basic data loader
    val_loader = DataLoader(dataset_val, collate_fn=Batch.collate(),
                            batch_size=16)  # basic data loader
    test_loader = DataLoader(dataset_test, collate_fn=Batch.collate(),
                             batch_size=16)  # basic data loader

    num_node_features = dataset_train.num_node_features
    num_classes = dataset_train.num_node_labels

    train(train_loader, val_loader,test_loader,
          args, num_node_features, num_classes, args.device)
