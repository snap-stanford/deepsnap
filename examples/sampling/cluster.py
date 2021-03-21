import copy
import torch
import random
import argparse
import numpy as np
import networkx as nx
import torch.nn as nn
import torch.nn.functional as F
import community as community_louvain

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
        epochs=150,
        hidden_size=64,
        num_layers=2,
        dropout=0.5,
        lr=0.005,
    )
    return parser.parse_args()

def preprocess(G, node_label_index, method="louvain"):
    graphs = []
    labeled_nodes = set(node_label_index.tolist())
    if method == "louvain":
        community_mapping = community_louvain.best_partition(G, resolution=10)
        communities = {}
        for node in community_mapping:
            comm = community_mapping[node]
            if comm in communities:
                communities[comm].add(node)
            else:
                communities[comm] = set([node])
        communities = communities.values()
    elif method == "bisection":
        communities = nx.algorithms.community.kernighan_lin_bisection(G)
    elif method == "greedy":
        communities = nx.algorithms.community.greedy_modularity_communities(G)

    for community in communities:
        nodes = set(community)
        subgraph = G.subgraph(nodes)
        # Make sure each subgraph has more than 10 nodes
        if subgraph.number_of_nodes() > 10:
            node_mapping = {node : i for i, node in enumerate(subgraph.nodes())}
            subgraph = nx.relabel_nodes(subgraph, node_mapping)
            # Get the id of the training set labeled node in the new graph
            train_label_index = []
            for node in labeled_nodes:
                if node in node_mapping:
                    # Append relabeled labeled node index
                    train_label_index.append(node_mapping[node])

            # Make sure the subgraph contains at least one training set labeled node
            if len(train_label_index) > 0:
                dg = Graph(subgraph)
                # Update node_label_index
                dg.node_label_index = torch.tensor(train_label_index, dtype=torch.long)
                graphs.append(dg)
    return graphs

def train(train_graphs, val_graphs, args, model, optimizer, mode="batch"):
    best_val = 0
    best_model = None
    accs = []
    graph_train = train_graphs[0]
    graph_train.to(args.device)
    for epoch in range(1, 1 + args.epochs):
        model.train()
        optimizer.zero_grad()
        if mode == "community":
            graph = random.choice(train_graphs)
            graph = graph.to(args.device)
            pred = model(graph, mode="all")
            pred = pred[graph.node_label_index]
            label = graph.node_label[graph.node_label_index]
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


    # Louvain
    graphs_train, graphs_val, graphs_test = \
        GraphDataset.pyg_to_graphs(pyg_dataset, verbose=True, fixed_split=True)
    graph_train = graphs_train[0]
    graph_val = graphs_val[0]
    graph_test = graphs_test[0]
    graphs = preprocess(graph_train.G, graph_train.node_label_index, method="louvain")
    print("Louvain: partition the graph in to {} communities".format(len(graphs)))
    avg_num_nodes = 0
    avg_num_edges = 0
    for graph in graphs:
        avg_num_nodes += graph.num_nodes
        avg_num_edges += graph.num_edges
    avg_num_nodes = int(avg_num_nodes / len(graphs))
    avg_num_edges = int(avg_num_edges / len(graphs))
    print("Each community has {} nodes in average".format(avg_num_nodes))
    print("Each community has {} edges in average".format(avg_num_edges))

    model = GNN(graph_train.num_node_features, args.hidden_size, graph_train.num_node_labels, args).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    louvain_best_model, louvain_accs = train(graphs, [graph_train, graph_val, graph_test], args, model, optimizer, mode="community")
    train_acc, val_acc, test_acc = test([graph_train, graph_val, graph_test], louvain_best_model)
    print('Best model:',
          f'Train: {100 * train_acc:.2f}%, '
          f'Valid: {100 * val_acc:.2f}% '
          f'Test: {100 * test_acc:.2f}%')


    # Bisection
    graphs_train, graphs_val, graphs_test = \
    GraphDataset.pyg_to_graphs(pyg_dataset, verbose=True, fixed_split=True)
    graph_train = graphs_train[0]
    graph_val = graphs_val[0]
    graph_test = graphs_test[0]
    graphs = preprocess(graph_train.G, graph_train.node_label_index, method="bisection")
    print("Bisection: partition the graph in to {} communities".format(len(graphs)))
    avg_num_nodes = 0
    avg_num_edges = 0
    for graph in graphs:
        avg_num_nodes += graph.num_nodes
        avg_num_edges += graph.num_edges
    avg_num_nodes = int(avg_num_nodes / len(graphs))
    avg_num_edges = int(avg_num_edges / len(graphs))
    print("Each community has {} nodes in average".format(avg_num_nodes))
    print("Each community has {} edges in average".format(avg_num_edges))

    model = GNN(graph_train.num_node_features, args.hidden_size, graph_train.num_node_labels, args).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    bisection_best_model, bisection_accs = train(graphs, [graph_train, graph_val, graph_test], args, model, optimizer, mode="community")
    train_acc, val_acc, test_acc = test([graph_train, graph_val, graph_test], bisection_best_model)
    print('Best model:',
          f'Train: {100 * train_acc:.2f}%, '
          f'Valid: {100 * val_acc:.2f}% '
          f'Test: {100 * test_acc:.2f}%')


    # Greedy
    graphs_train, graphs_val, graphs_test = \
    GraphDataset.pyg_to_graphs(pyg_dataset, verbose=True, fixed_split=True)
    graph_train = graphs_train[0]
    graph_val = graphs_val[0]
    graph_test = graphs_test[0]
    graphs = preprocess(graph_train.G, graph_train.node_label_index, method="greedy")
    print("Greedy: partition the graph in to {} communities".format(len(graphs)))
    avg_num_nodes = 0
    avg_num_edges = 0
    for graph in graphs:
        avg_num_nodes += graph.num_nodes
        avg_num_edges += graph.num_edges
    avg_num_nodes = int(avg_num_nodes / len(graphs))
    avg_num_edges = int(avg_num_edges / len(graphs))
    print("Each community has {} nodes in average".format(avg_num_nodes))
    print("Each community has {} edges in average".format(avg_num_edges))

    model = GNN(graph_train.num_node_features, args.hidden_size, graph_train.num_node_labels, args).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    bisection_best_model, bisection_accs = train(graphs, [graph_train, graph_val, graph_test], args, model, optimizer, mode="community")
    train_acc, val_acc, test_acc = test([graph_train, graph_val, graph_test], bisection_best_model)
    print('Best model:',
          f'Train: {100 * train_acc:.2f}%, '
          f'Valid: {100 * val_acc:.2f}% '
          f'Test: {100 * test_acc:.2f}%')


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