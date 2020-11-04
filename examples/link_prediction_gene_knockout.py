import argparse
import copy
import csv
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import networkx as nx
import numpy as np
import sklearn.metrics as metrics
import torch_geometric.transforms as T
import torch_geometric.nn as pyg_nn
from torch.utils.data import DataLoader
from deepsnap.graph import Graph
from deepsnap.dataset import GraphDataset
from deepsnap.batch import Batch


def arg_parse():
    parser = argparse.ArgumentParser(description='Link pred arguments.')
    parser.add_argument('--device', type=str,
                        help='CPU / GPU device.')
    parser.add_argument('--data_path', type=str,
                        help='Path to cmap data file.')
    parser.add_argument('--ppi_path', type=str,
                        help='Path to ppi data file.')
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
        data_path='data/rex_cmap_data.csv',
        ppi_path='data/pc3_biogrid_ppi.csv',
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
            self.conv1 = MlpMessageConv(
                input_dim, args.hidden_dim, edge_feat_dim
            )
            self.conv2 = MlpMessageConv(
                args.hidden_dim, args.hidden_dim, edge_feat_dim
            )
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

        nodes_first = torch.index_select(
            x, 0, graph.edge_label_index[0, :].long()
        )
        nodes_second = torch.index_select(
            x, 0, graph.edge_label_index[1, :].long()
        )

        # directed / asym link pred
        pred = self.lp_mlp(torch.cat((nodes_first, nodes_second), dim=-1))
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
                nn.ReLU()
            )
        self.message_mlp = nn.Sequential(
            nn.Linear(in_channels * 2 + edge_dim, out_channels),
        )

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


def cmap_transform(graph, num_edge_types, input_dim=5):
    # get nx graph
    G = graph.G
    for v in G.nodes:
        G.nodes[v]['node_feature'] = torch.ones(input_dim)

    for u, v, edge_key in G.edges:
        l = G[u][v][edge_key].pop('edge_de', None)
        e_feat = torch.zeros(num_edge_types)
        e_feat[l] = 1.

        # here both feature and label are relation types
        G[u][v][edge_key]['edge_feature'] = e_feat
        G[u][v][edge_key]['edge_label'] = l

    graph.G = G
    return graph


def train(model, dataloaders, optimizer, args, writer):
    # training loop
    val_max = -np.inf
    best_model = model
    t_accu = []
    v_accu = []
    # e_accu = []
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

            # log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
            log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}'
            accs, _ = test(model, dataloaders, args)
            t_accu.append(accs['train'])
            v_accu.append(accs['val'])
            # e_accu.append(accs['test'])

            # print(log.format(epoch, accs['train'], accs['val'], accs['test']))
            print(log.format(epoch, accs['train'], accs['val']))
            writer.add_scalar('Loss/train', accs['train'], iter_i)
            writer.add_scalar('Loss/val', accs['val'], iter_i)
            # writer.add_scalar('Loss/test', accs['test'], iter_i)
            if val_max < accs['val']:
                val_max = accs['val']
                best_model = copy.deepcopy(model)
            print('Time: ', time.time() - start_t)

    # log = 'Best, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    log = 'Best, Train: {:.4f}, Val: {:.4f}'
    accs, _ = test(best_model, dataloaders, args)
    # print(log.format(accs['train'], accs['val'], accs['test']))
    print(log.format(accs['train'], accs['val']))


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


def read_and_split_cmap_data(path, split_ratio=[0.8, 0.2]):
    nxG_train = nx.DiGraph()
    nxG_test = nx.DiGraph()

    train_ratio = split_ratio[0]

    with open(path, newline='') as csvfile:
        reader = csv.reader(
            csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC
        )
        next(reader)
        for row in reader:
            if row[2] == -1:
                de = 0
            elif row[2] == 1:
                de = 1
            if np.random.rand() < train_ratio:
                nxG_train.add_edge(
                    row[0], row[1], edge_de=de,
                    edge_dose=row[4], edge_pert_time=row[5]
                )
            else:
                nxG_test.add_edge(
                    row[0], row[1], edge_de=de,
                    edge_dose=row[4], edge_pert_time=row[5]
                )
    nxG_train.add_nodes_from(nxG_test)
    nxG_test.add_nodes_from(nxG_train)
    return nxG_train, nxG_test


def read_cmap_data(path, nxG=None):
    knockout_nodes = set()
    if nxG is None:
        nxG = nx.DiGraph()
    with open(path, newline='') as csvfile:
        reader = csv.reader(
            csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC
        )
        next(reader)
        prev_exp_id = None
        skip_exp = False
        for row in reader:
            # assumes that experiment ids are consecutive in csv
            # remove experiment on the same knockout gene
            if not row[3] == prev_exp_id:
                # encounter new experiment
                prev_exp_id = row[3]
                if row[0] in knockout_nodes:
                    skip_exp = True
                    continue
                else:
                    skip_exp = False
                    knockout_nodes.add(row[0])
            else:
                # continue current experiment
                if skip_exp:
                    continue
            if row[2] == -1:
                de = 0
            elif row[2] == 1:
                de = 1
            nxG.add_edge(
                row[0], row[1], edge_type=1,
                edge_de=de, edge_dose=row[4], edge_pert_time=row[5]
            )

    return nxG, knockout_nodes


def read_ppi_data(path):
    nxG = nx.MultiDiGraph()

    with open(path, newline='') as csvfile:
        reader = csv.reader(
            csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC
        )
        next(reader)
        for row in reader:
            # default field values for ppi edges (edge_type 0)
            if nxG.has_edge(row[0], row[1]):
                raise RuntimeError('PPI has repeated edges')
            nxG.add_edge(
                row[0], row[1], edge_type=0,
                edge_de=0, edge_dose=0, edge_pert_time=0
            )
            # self loop
            if row[0] == row[1]:
                continue
            nxG.add_edge(
                row[1], row[0], edge_type=0,
                edge_de=0, edge_dose=0, edge_pert_time=0
            )
    return nxG


def main():
    writer = SummaryWriter()
    args = arg_parse()

    edge_train_mode = args.mode
    print('edge train mode: {}'.format(edge_train_mode))

    ppi_graph = read_ppi_data(args.ppi_path)

    mode = 'mixed'
    if mode == 'ppi':
        message_passing_graph = ppi_graph
        cmap_graph, knockout_nodes = read_cmap_data(args.data_path)
    elif mode == 'mixed':
        message_passing_graph, knockout_nodes = (
            read_cmap_data(args.data_path, ppi_graph)
        )

    print('Each node has gene ID. Example: ', message_passing_graph.nodes['ADPGK'])
    print('Each edge has de direction. Example', message_passing_graph['ADPGK']['IL1B'])
    print('Total num edges: ', message_passing_graph.number_of_edges())

    # disjoint edge label
    disjoint_split_ratio = 0.1
    val_ratio = 0.1
    disjoint_edge_label_index = []
    val_edges = []

    # newly edited
    train_edges = []
    for u in knockout_nodes:
        rand_num = np.random.rand()
        if rand_num < disjoint_split_ratio:
            # add all edges (cmap only) into edge label index
            # cmap is not a multigraph
            disjoint_edge_label_index.extend(
                [
                    (u, v, edge_key)
                    for v in message_passing_graph.successors(u)
                    for edge_key in message_passing_graph[u][v]
                    if message_passing_graph[u][v][edge_key]['edge_type'] == 1
                ]
            )

            train_edges.extend(
                [
                    (u, v, edge_key)
                    for v in message_passing_graph.successors(u)
                    for edge_key in message_passing_graph[u][v]
                    if message_passing_graph[u][v][edge_key]['edge_type'] == 1
                ]
            )
        elif rand_num < disjoint_split_ratio + val_ratio:
            val_edges.extend(
                [
                    (u, v, edge_key)
                    for v in message_passing_graph.successors(u)
                    for edge_key in message_passing_graph[u][v]
                    if message_passing_graph[u][v][edge_key]['edge_type'] == 1
                ]
            )
        else:
            train_edges.extend(
                [
                    (u, v, edge_key)
                    for v in message_passing_graph.successors(u)
                    for edge_key in message_passing_graph[u][v]
                    if message_passing_graph[u][v][edge_key]['edge_type'] == 1
                ]
            )

    print('Num edges to predict: ', len(disjoint_edge_label_index))
    print('Num edges in val: ', len(val_edges))
    print('Num edges in train: ', len(train_edges))

    graph = Graph(
        message_passing_graph,
        custom={
            "general_splits": [
                train_edges,
                val_edges
            ],
            "disjoint_split": disjoint_edge_label_index,
            "task": "link_pred"
        }
    )
    graphs = [graph]
    graphDataset = GraphDataset(
        graphs,
        task="link_pred",
        edge_train_mode="disjoint"
    )

    # Transform dataset
    # de direction (currently using homogeneous graph)
    num_edge_types = 2

    graphDataset = graphDataset.apply_transform(
        cmap_transform, num_edge_types=num_edge_types, deep_copy=False
    )
    print('Number of node features: {}'.format(graphDataset.num_node_features))

    # split dataset
    dataset = {}
    dataset['train'], dataset['val'] = graphDataset.split(transductive=True)

    # sanity check
    print(f"dataset['train'][0].edge_label_index.shape[1]: {dataset['train'][0].edge_label_index.shape[1]}")
    print(f"dataset['val'][0].edge_label_index.shape[1]: {dataset['val'][0].edge_label_index.shape[1]}")
    print(f"len(list(dataset['train'][0].G.edges)): {len(list(dataset['train'][0].G.edges))}")
    print(f"len(list(dataset['val'][0].G.edges)): {len(list(dataset['val'][0].G.edges))}")
    print(f"list(dataset['train'][0].G.edges)[:10]: {list(dataset['train'][0].G.edges)[:10]}")
    print(f"list(dataset['val'][0].G.edges)[:10]: {list(dataset['val'][0].G.edges)[:10]}")


    # node feature dimension
    input_dim = dataset['train'].num_node_features
    edge_feat_dim = dataset['train'].num_edge_features
    num_classes = dataset['train'].num_edge_labels
    print(
        'Node feature dim: {}; edge feature dim: {}; num classes: {}.'.format(
            input_dim, edge_feat_dim, num_classes
        )
    )

    # relation type is both used for edge features and edge labels
    model = Net(input_dim, edge_feat_dim, num_classes, args).to(args.device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001, weight_decay=5e-3
    )
    follow_batch = []  # e.g., follow_batch = ['edge_index']

    dataloaders = {
        split: DataLoader(
            ds, collate_fn=Batch.collate(follow_batch),
            batch_size=1, shuffle=(split == 'train')
        )
        for split, ds in dataset.items()
    }
    print('Graphs after split: ')
    for key, dataloader in dataloaders.items():
        for batch in dataloader:
            print(key, ': ', batch)

    train(model, dataloaders, optimizer, args, writer=writer)


if __name__ == '__main__':
    main()
