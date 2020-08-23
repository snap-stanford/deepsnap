import sys
sys.path.append('../')

import argparse
import copy
import math
import time

import numpy as np
from sklearn.metrics import *
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
import torch_geometric.nn as pyg_nn

from deepsnap.dataset import GraphDataset
from deepsnap.batch import Batch

import pdb

def arg_parse():
    parser = argparse.ArgumentParser(description='Link pred arguments.')
    parser.add_argument('--device', type=str,
                        help='CPU / GPU device.')
    parser.add_argument('--epochs', type=int,
                        help='Number of epochs to train.')
    parser.add_argument('--mode', type=str,
                        help='Link prediction mode. Disjoint or all.')
    parser.add_argument('--model', type=str,
                        help='GCN or Spline.')
    parser.add_argument('--edge_message_ratio', type=float,
                        help='Ratio of edges used for message-passing (only in disjoint mode).')
    parser.add_argument('--multigraph', action='store_true',
                        help='Example of multi-graph link pred (by copying CORA 10 times).')
    parser.add_argument('--batch_size', type=int,
                        help='Batch size for multi-graph link pred. Always 1 for single-graph case.')
    parser.add_argument('--hidden_dim', type=int,
                        help='Hidden dimension of GNN.')

    parser.set_defaults(
            device='cuda:0', 
            epochs=500,
            mode='all',
            model='GCN',
            edge_message_ratio=0.6,
            multigraph=False,
            batch_size=1,
            hidden_dim=16,
    )
    return parser.parse_args()

class Net(torch.nn.Module):
    def __init__(self, input_dim, num_classes, args):
        super(Net, self).__init__()
        self.model = args.model
        if self.model == 'GCN':
            self.conv1 = pyg_nn.GCNConv(input_dim, args.hidden_dim)
            self.conv2 = pyg_nn.GCNConv(args.hidden_dim, num_classes)
        elif self.model == 'Spline':
            self.conv1 = pyg_nn.SplineConv(input_dim, args.hidden_dim, dim=1, kernel_size=2)
            self.conv2 = pyg_nn.SplineConv(args.hidden_dim, num_classes, dim=1, kernel_size=2)
        else:
            raise ValueError('unknown conv')
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def forward(self, graph):
        # (batch of) graph object(s) containing all the tensors we want
        x = F.dropout(graph.node_feature, p=0.2, training=self.training)
        x = F.relu(self._conv_op(self.conv1, x, graph))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self._conv_op(self.conv2, x, graph)

        nodes_first = torch.index_select(x, 0, graph.edge_label_index[0,:].long())
        nodes_second = torch.index_select(x, 0, graph.edge_label_index[1,:].long())
        pred = torch.sum(nodes_first * nodes_second, dim=-1)
        #pred = torch.nn.CosineSimilarity(dim=-1)(nodes_first, nodes_second)
        #pred = torch.sigmoid(pred)
        return pred
    
    def _conv_op(self, conv, x, graph):
        if self.model == 'GCN':
            return conv(x, graph.edge_index)
        elif self.model == 'spline':
            return conv(x, graph.edge_index, graph.edge_feature)

    def loss(self, pred, link_label):
        return self.loss_fn(pred, link_label)


def train(model, dataloaders, optimizer, args, scheduler=None):

    # training loop
    val_max = -math.inf
    best_model = model
    t_accu = []
    v_accu = []
    e_accu = []
    for epoch in range(1, args.epochs):
        for iter_i, batch in enumerate(dataloaders['train']):
            start_t = time.time()
            batch.to(args.device)
            model.train()
            optimizer.zero_grad()
            pred = model(batch)
            loss = model.loss(pred, batch.edge_label.type(pred.dtype))
            print('loss: ', loss.item())
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
            accs, _ = test(model, dataloaders, args)
            t_accu.append(accs['train'])
            v_accu.append(accs['val'])
            e_accu.append(accs['test'])

            print(log.format(epoch, accs['train'], accs['val'], accs['test']))
            if val_max < accs['val']:
                val_max = accs['val']
                best_model = copy.deepcopy(model)
            print('Time: ', time.time() - start_t)

    log = 'Best, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    accs, _ = test(best_model, dataloaders, args)
    print(log.format(accs['train'], accs['val'], accs['test']))

def test(model, dataloaders, args, max_train_batches=1):
    model.eval()
    accs = {}
    losses = {}
    for mode, dataloader in dataloaders.items():
        acc = 0
        loss = 0
        num_batches = 0
        for batch in dataloader:
            batch.to(args.device)
            pred = model(batch)
            # only 1 graph in dataset. In general needs aggregation
            loss += model.loss(pred, batch.edge_label.type(pred.dtype)).cpu().data.numpy()
            acc += roc_auc_score(batch.edge_label.flatten().cpu().numpy(), 
                                pred.flatten().data.cpu().numpy())
            num_batches += 1
            if mode == 'train' and num_batches >= max_train_batches:
                # do not eval on the entire training set for efficiency
                break
        accs[mode] = acc / num_batches
        losses[mode] = loss / num_batches
    return accs, losses

def main():
    args = arg_parse()

    pyg_dataset = Planetoid('./cora', 'Cora', transform=T.TargetIndegree())
    
    # the input that we assume users have
    edge_train_mode = args.mode
    print('edge train mode: {}'.format(edge_train_mode))

    graphs = GraphDataset.pyg_to_graphs(pyg_dataset)
    if args.multigraph:
        graphs = [copy.deepcopy(graphs[0]) for _ in range(10)]

    dataset = GraphDataset(graphs, 
                           task='link_pred', 
                           edge_message_ratio=args.edge_message_ratio, 
                           edge_train_mode=edge_train_mode)
    print('Initial dataset: {}'.format(dataset))

    # split dataset
    datasets = {}
    datasets['train'], datasets['val'], datasets['test']= dataset.split(
            transductive=not args.multigraph, split_ratio=[0.85, 0.05, 0.1])

    print('after split')
    print('Train message-passing graph: {} nodes; {} edges.'.format(
            datasets['train'][0].G.number_of_nodes(),
            datasets['train'][0].G.number_of_edges()))
    print('Val message-passing graph: {} nodes; {} edges.'.format(
            datasets['val'][0].G.number_of_nodes(),
            datasets['val'][0].G.number_of_edges()))
    print('Test message-passing graph: {} nodes; {} edges.'.format(
            datasets['test'][0].G.number_of_nodes(),
            datasets['test'][0].G.number_of_edges()))


    # node feature dimension
    input_dim = datasets['train'].num_node_features
    # link prediction needs 2 classes (0, 1)
    num_classes = datasets['train'].num_edge_labels

    model = Net(input_dim, num_classes, args).to(args.device)
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    follow_batch = [] # e.g., follow_batch = ['edge_index']

    dataloaders = {split: DataLoader(
            ds, collate_fn=Batch.collate(follow_batch), 
            batch_size=args.batch_size, shuffle=(split=='train'))
            for split, ds in datasets.items()}
    print('Graphs after split: ')
    for key, dataloader in dataloaders.items():
        for batch in dataloader:
            print(key, ': ', batch)

    train(model, dataloaders, optimizer, args, scheduler=scheduler)

if __name__ == '__main__':
    main()

