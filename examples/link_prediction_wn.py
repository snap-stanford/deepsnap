import copy
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx 
import numpy as np
import sklearn.metrics as metrics
import torch_geometric.transforms as T
import torch_geometric.nn as pyg_nn
from torch.utils.data import DataLoader
from deepsnap.dataset import GraphDataset
from deepsnap.batch import Batch

def arg_parse():
    parser = argparse.ArgumentParser(description='Link pred arguments.')
    parser.add_argument('--device', type=str,
                        help='CPU / GPU device.')
    parser.add_argument('--data_path', type=str,
                        help='Path to wordnet nx gpickle file.')
    parser.add_argument('--epochs', type=int,
                        help='Number of epochs to train.')
    parser.add_argument('--mode', type=str,
                        help='Link prediction mode. Disjoint or all.')
    parser.add_argument('--model', type=str,
                        help='MlpMessage.')
    parser.add_argument('--edge_message_ratio', type=float,
                        help='Ratio of edges used for message-passing (only in disjoint mode).')
    parser.add_argument('--neg_sampling_ratio', type=float,
                        help='Ratio of the number of negative examples to the number of positive examples')
    parser.add_argument('--hidden_dim', type=int,
                        help='Hidden dimension of GNN.')

    parser.set_defaults(
            device='cuda:0',
            data_path='data/WN18.gpickle',
            epochs=500,
            mode='disjoint',
            model='MlpMessage',
            edge_message_ratio=0.8,
            neg_sampling_ratio=1.0,
            hidden_dim=16
    )
    return parser.parse_args()


class Net(torch.nn.Module):
    def __init__(self, input_dim, edge_feat_dim, num_classes, args):
        super(Net, self).__init__()
        self.model = args.model
        if self.model == 'MlpMessage':
            self.conv1 = MlpMessageConv(input_dim, args.hidden_dim, edge_feat_dim)
            self.conv2 = MlpMessageConv(args.hidden_dim, args.hidden_dim, edge_feat_dim)
        else:
            raise ValueError('unknown conv')
        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.lp_mlp = nn.Linear(args.hidden_dim * 2, num_classes)

    def forward(self, graph):
        # (batch of) graph object(s) containing all the tensors we want
        x = F.dropout(graph.node_feature, p=0.2, training=self.training)
        x = F.relu(self._conv_op(self.conv1, x, graph))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self._conv_op(self.conv2, x, graph)

        nodes_first = torch.index_select(x, 0, graph.edge_label_index[0,:].long())
        nodes_second = torch.index_select(x, 0, graph.edge_label_index[1,:].long())
        pred = self.lp_mlp(torch.cat((nodes_first, nodes_second), dim=-1))
        #pred = torch.nn.CosineSimilarity(dim=-1)(nodes_first, nodes_second)
        return pred

    def _conv_op(self, conv_op, x, graph):
        return conv_op(x, graph.edge_index, graph.edge_feature)
    
    def loss(self, pred, link_label):
        return self.loss_fn(pred, link_label)

    def inference(self, pred):
        return torch.argmax(pred, -1)


class MlpMessageConv(pyg_nn.MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dim, pre_conv=True):
        super(MlpMessageConv, self).__init__(aggr='add')
        self.pre_conv = None
        if pre_conv:
            self.pre_conv = nn.Sequential(
                    nn.Linear(in_channels, in_channels),
                    nn.ReLU())
        self.message_mlp = nn.Sequential(
                nn.Linear(in_channels * 2 + edge_dim, out_channels),)

    def forward(self, x, edge_index, edge_feature):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        if self.pre_conv is not None:
            x = self.pre_conv(x)
        return self.propagate(
            edge_index,
            x=x,
            edge_feature=edge_feature
        )

    def message(self, x_i, x_j, edge_feature):
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]

        tmp = torch.cat([x_i, x_j, edge_feature], dim=1)  # tmp has shape [E, 2 * in_channels]
        return self.message_mlp(tmp)

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]

        return aggr_out


def WN_transform(graph, num_edge_types, input_dim=5):
    # get nx graph
    G = graph.G
    for v in G.nodes:
        G.nodes[v]['node_feature'] = torch.ones(input_dim)

    for u, v, edge_key in G.edges:
        l = G[u][v][edge_key]['e_label']
        e_feat = torch.zeros(num_edge_types)
        e_feat[l] = 1.
        # here both feature and label are relation types
        G[u][v][edge_key]['edge_feature'] = e_feat
        G[u][v][edge_key]['edge_label'] = l
    # optionally return the graph or G object
    graph.G = G
    return graph


def train(model, dataloaders, optimizer, args):
    # training loop
    val_max = -np.inf
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
            loss = model.loss(pred, batch.edge_label)
            print('loss: ', loss.item())
            loss.backward()
            optimizer.step()

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
            loss += model.loss(pred, batch.edge_label).cpu().data.numpy()
            acc += metrics.f1_score(
                    batch.edge_label.cpu().numpy(),
                    model.inference(pred).cpu().numpy(),
                    average='micro')
            num_batches += 1
            if mode == 'train' and num_batches >= max_train_batches:
                # do not eval on the entire training set for efficiency
                break
        accs[mode] = acc / num_batches
        losses[mode] = loss / num_batches
    return accs, losses


def main():
    args = arg_parse()

    edge_train_mode = args.mode
    print('edge train mode: {}'.format(edge_train_mode))

    WN_graph = nx.read_gpickle(args.data_path)
    print('Each node has node ID (n_id). Example: ', WN_graph.nodes[0])
    print('Each edge has edge ID (id) and categorical label (e_label). Example: ', WN_graph[0][5871])

    # Since both feature and label are relation types,
    # Only the disjoint mode would make sense
    dataset = GraphDataset(
        [WN_graph], task='link_pred', 
        edge_train_mode=edge_train_mode,
        edge_message_ratio=args.edge_message_ratio,
        edge_negative_sampling_ratio=args.neg_sampling_ratio
    )

    # find num edge types
    max_label = 0
    labels = []
    for u, v, edge_key in WN_graph.edges:
        l = WN_graph[u][v][edge_key]['e_label']
        if not l in labels:
            labels.append(l)
    # labels are consecutive (0-17)
    num_edge_types = len(labels)

    print('Pre-transform: ', dataset[0])
    dataset = dataset.apply_transform(WN_transform, num_edge_types=num_edge_types,
                            deep_copy=False)
    print('Post-transform: ', dataset[0])
    print('Initial data: {} nodes; {} edges.'.format(
            dataset[0].G.number_of_nodes(),
            dataset[0].G.number_of_edges()))
    print('Number of node features: {}'.format(dataset.num_node_features))

    # split dataset
    datasets = {}
    datasets['train'], datasets['val'], datasets['test']= dataset.split(
            transductive=True, split_ratio=[0.8, 0.1, 0.1])

    print('After split:')
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
    edge_feat_dim = datasets['train'].num_edge_features
    num_classes = datasets['train'].num_edge_labels
    print('Node feature dim: {}; edge feature dim: {}; num classes: {}.'.format(
            input_dim, edge_feat_dim, num_classes))

    # relation type is both used for edge features and edge labels
    model = Net(input_dim, edge_feat_dim, num_classes, args).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-3)
    follow_batch = [] # e.g., follow_batch = ['edge_index']

    dataloaders = {split: DataLoader(
            ds, collate_fn=Batch.collate(follow_batch), 
            batch_size=1, shuffle=(split=='train'))
            for split, ds in datasets.items()}
    print('Graphs after split: ')
    for key, dataloader in dataloaders.items():
        for batch in dataloader:
            print(key, ': ', batch)

    train(model, dataloaders, optimizer, args)

if __name__ == '__main__':
    main()
