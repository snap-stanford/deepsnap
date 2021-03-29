import copy
import torch
import random
import argparse
import numpy as np
import networkx as nx
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import SAGEConv
from torch.utils.data import DataLoader
from torch_geometric.datasets import Planetoid
from torch.nn import Sequential, Linear, ReLU
from deepsnap.dataset import GraphDataset
from deepsnap.graph import Graph
from model import *

def arg_parse():
    parser = argparse.ArgumentParser(description='Sampling arguments.')

    parser.add_argument('--device', type=str,
                        help='CPU / GPU device.')
    parser.add_argument('--epochs', type=int,
                        help='Number of epochs to train.')
    parser.add_argument('--num_layers', type=int,
                        help='Number of GNN layers.')
    parser.add_argument('--hidden_size', type=int,
                        help='GNN layer hidden dimension size.')
    parser.add_argument('--dropout', type=float,
                        help='The dropout ratio.')
    parser.add_argument('--lr', type=float,
                        help='Learning rate.')

    parser.set_defaults(
        device='cuda:0',
        epochs=50,
        hidden_size=64,
        num_layers=2,
        dropout=0.5,
        lr=0.005,
    )
    return parser.parse_args()

def sample_neighbors(nodes, G, ratio, all_nodes):
    # This fuction takes a set of nodes, a NetworkX graph G and neighbor sampling ratio.
    # It will return sampled neighbors (unioned with input nodes) and edges between 
    neighbors = set()
    edges = []
    for node in nodes:
        neighbors_list = list(nx.neighbors(G, node))

        # We only sample the (ratio * number of neighbors) neighbors
        num = int(len(neighbors_list) * ratio)
        if num > 0:
            # Random shuffle the neighbors
            random.shuffle(neighbors_list)
            neighbors_list = neighbors_list[:num]
            for neighbor in neighbors_list:
                # Add neighbors
                neighbors.add(neighbor)
                edges.append((neighbor, node))
    return neighbors, neighbors.union(all_nodes), edges

def nodes_to_tensor(nodes):
    # This function transform a set of nodes to node index tensor
    node_label_index = torch.tensor(list(nodes), dtype=torch.long)
    return node_label_index

def edges_to_tensor(edges):
    # This function transform a set of edges to edge index tensor
    edge_index = torch.tensor(list(edges), dtype=torch.long)
    edge_index = torch.cat([edge_index, torch.flip(edge_index, [1])], dim=0)
    edge_index = edge_index.permute(1, 0)
    return edge_index

def relable(nodes, labeled_nodes, edges_list):
    # Relable the nodes, labeled_nodes and edges_list
    relabled_edges_list = []
    sorted_nodes = sorted(nodes)
    node_mapping = {node : i for i, node in enumerate(sorted_nodes)}
    for orig_edges in edges_list:
        relabeled_edges = []
        for edge in orig_edges:
            relabeled_edges.append((node_mapping[edge[0]], node_mapping[edge[1]]))
        relabled_edges_list.append(relabeled_edges)
    relabeled_labeled_nodes = [node_mapping[node] for node in labeled_nodes]
    relabeled_nodes = [node_mapping[node] for node in nodes]
    return relabled_edges_list, relabeled_nodes, relabeled_labeled_nodes

def neighbor_sampling(graph, K=2, ratios=(0.1, 0.1, 0.1)):
    # This function takes a DeepSNAP graph, K the number of GNN layers, and neighbor 
    # sampling ratios for each layer. This function returns relabeled node feature, 
    # edge indices and node_label_index

    assert K + 1 == len(ratios)

    labeled_nodes = graph.node_label_index.tolist()
    random.shuffle(labeled_nodes)
    num = int(len(labeled_nodes) * ratios[-1])
    if num > 0:
        labeled_nodes = labeled_nodes[:num]
    nodes_list = [set(labeled_nodes)]
    edges_list = []
    all_nodes = labeled_nodes
    for k in range(K):
        # Get nodes and edges from the previous layer
        nodes, all_nodes, edges = \
            sample_neighbors(nodes_list[-1], graph.G, ratios[len(ratios) - k - 2], all_nodes)
        nodes_list.append(nodes)
        edges_list.append(edges)
    
    # Reverse the lists
    nodes_list.reverse()
    edges_list.reverse()

    relabled_edges_list, relabeled_all_nodes, relabeled_labeled_nodes = \
        relable(all_nodes, labeled_nodes, edges_list)

    node_index = nodes_to_tensor(relabeled_all_nodes)
    # All node features that will be used
    node_feature = graph.node_feature[node_index]
    edge_indices = [edges_to_tensor(edges) for edges in relabled_edges_list]
    node_label_index = nodes_to_tensor(relabeled_labeled_nodes)
    log = "Sampled {} nodes, {} edges, {} labeled nodes"
    print(log.format(node_feature.shape[0], edge_indices[0].shape[1] // 2, node_label_index.shape[0]))
    return node_feature, edge_indices, node_label_index

def train(train_graphs, val_graphs, args, model, optimizer, mode="batch"):
    best_val = 0
    best_model = None
    accs = []
    graph_train = train_graphs[0]
    graph_train.to(args.device)
    for epoch in range(1, 1 + args.epochs):
        model.train()
        optimizer.zero_grad()
        if mode == "batch":
            node_feature, edge_indices, node_label_index = neighbor_sampling(graph_train, args.num_layers, args.ratios)
            node_feature = node_feature.to(args.device)
            node_label_index = node_label_index.to(args.device)
            for i in range(len(edge_indices)):
                edge_indices[i] = edge_indices[i].to(args.device)
            pred = model([edge_indices, node_feature])
            pred = pred[node_label_index]
            label = graph_train.node_label[node_label_index]
        else:
            pred = model(graph_train, mode="all")
            label = graph_train.node_label
            pred = pred[graph_train.node_label_index]
        loss = F.nll_loss(pred, label)
        loss.backward()
        optimizer.step()

        train_acc, val_acc, test_acc = test(val_graphs, model)
        accs.append((train_acc, val_acc, test_acc))
        if val_acc > best_val:
            best_val = val_acc
            best_model = copy.deepcopy(model)
        print(f'Epoch: {epoch:02d}, '
              f'Loss: {loss:.4f}, '
              f'Train: {100 * train_acc:.2f}%, '
              f'Valid: {100 * val_acc:.2f}% '
              f'Test: {100 * test_acc:.2f}%')
    return best_model, accs

def test(graphs, model):
    model.eval()
    accs = []
    for graph in graphs:
        graph = graph.to(args.device)
        pred = model(graph, mode="all")
        label = graph.node_label
        pred = pred[graph.node_label_index].max(1)[1]
        acc = pred.eq(label).sum().item()
        acc /= len(label)
        accs.append(acc)
    return accs

if __name__ == '__main__':
    args = arg_parse()
    pyg_dataset = Planetoid('./planetoid', 'Cora')


    # Full batch
    print("Full batch training")
    graphs_train, graphs_val, graphs_test = \
    GraphDataset.pyg_to_graphs(pyg_dataset, verbose=True, fixed_split=True)
    graph_train = graphs_train[0]
    graph_val = graphs_val[0]
    graph_test = graphs_test[0]
    model = GNN(graph_train.num_node_features, args.hidden_size, graph_train.num_node_labels, args).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    graphs = [graph_train, graph_val, graph_test]
    all_best_model, all_accs = train(graphs, graphs, args, model, optimizer, mode="all")
    train_acc, val_acc, test_acc = test([graph_train, graph_val, graph_test], all_best_model)
    print('Best model:',
          f'Train: {100 * train_acc:.2f}%, '
          f'Valid: {100 * val_acc:.2f}% '
          f'Test: {100 * test_acc:.2f}%')


    # 0.8
    args.ratios = (0.8, 0.8, 1)
    print("Sampling ratios: {}".format(args.ratios))
    graphs_train, graphs_val, graphs_test = \
        GraphDataset.pyg_to_graphs(pyg_dataset, verbose=True, fixed_split=True)
    graph_train = graphs_train[0]
    graph_val = graphs_val[0]
    graph_test = graphs_test[0]
    model = GNN(graph_train.num_node_features, args.hidden_size, graph_train.num_node_labels, args).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    graphs = [graph_train, graph_val, graph_test]
    batch_best_model, batch_accs = train(graphs, graphs, args, model, optimizer)
    train_acc, val_acc, test_acc = test([graph_train, graph_val, graph_test], batch_best_model)
    print('Best model:',
          f'Train: {100 * train_acc:.2f}%, '
          f'Valid: {100 * val_acc:.2f}% '
          f'Test: {100 * test_acc:.2f}%')


    # 0.3
    args.ratios = (0.3, 0.3, 1)
    print("Sampling ratios: {}".format(args.ratios))
    graphs_train, graphs_val, graphs_test = \
        GraphDataset.pyg_to_graphs(pyg_dataset, verbose=True, fixed_split=True)
    graph_train = graphs_train[0]
    graph_val = graphs_val[0]
    graph_test = graphs_test[0]
    model = GNN(graph_train.num_node_features, args.hidden_size, graph_train.num_node_labels, args).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    graphs = [graph_train, graph_val, graph_test]
    batch_best_model, batch_accs = train(graphs, graphs, args, model, optimizer)
    train_acc, val_acc, test_acc = test([graph_train, graph_val, graph_test], batch_best_model)
    print('Best model:',
          f'Train: {100 * train_acc:.2f}%, '
          f'Valid: {100 * val_acc:.2f}% '
          f'Test: {100 * test_acc:.2f}%')