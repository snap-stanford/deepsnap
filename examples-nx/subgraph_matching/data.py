import os
import pickle
import random

from deepsnap.batch import Batch
from deepsnap.dataset import GraphDataset
import networkx as nx
import numpy as np
from sklearn.manifold import TSNE
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import DataLoader
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.datasets import TUDataset, PPI, QM9
import torch_geometric.utils as pyg_utils
import torch_geometric.nn as pyg_nn
from tqdm import tqdm
import queue

import utils

def load_dataset(name):
    def add_feats(graph):
        for v in graph.G.nodes:
            graph.G.nodes[v]["node_feature"] = torch.ones(1)
        return graph
    task = "graph"
    if name == "enzymes":
        dataset = TUDataset(root="/tmp/ENZYMES", name="ENZYMES")
    elif name == "cox2":
        dataset = TUDataset(root="/tmp/cox2", name="COX2")
    elif name == "imdb-binary":
        dataset = TUDataset(root="/tmp/IMDB-BINARY", name="IMDB-BINARY")

    if task == "graph":
        dataset = GraphDataset(GraphDataset.pyg_to_graphs(dataset))
        # add blank features for imdb-binary, which doesn't have node labels
        if name == "imdb-binary":
            dataset = dataset.apply_transform(add_feats)
        dataset = dataset.apply_transform(lambda g:
            g.G.subgraph(max(nx.connected_components(g.G), key=len)))
        dataset = dataset.filter(lambda g: len(g.G) >= 6)
        train, test = dataset.split(split_ratio=[0.8, 0.2])
    return train, test, task

class DataSource:
    def __init__(self, dataset_name, min_size=5, max_size=20):
        self.min_size = min_size
        self.max_size = max_size
        self.train, self.test, _ = load_dataset(dataset_name)

    def gen_batch(self, batch_target, batch_neg_target, batch_neg_query,
        train):

        def sample_subgraph(graph, offset=0, use_precomp_sizes=False):
            if use_precomp_sizes:
                size = graph.G.graph["subgraph_size"]
            else:
                size = random.randint(self.min_size + offset,
                    len(graph.G) - 1 + offset)
            start_node = random.choice(list(graph.G.nodes))
            neigh = [start_node]
            frontier = list(set(graph.G.neighbors(start_node)) - set(neigh))
            visited = set([start_node])
            while len(neigh) < size and frontier:
                new_node = random.choice(list(frontier))
                assert new_node not in neigh
                neigh.append(new_node)
                visited.add(new_node)
                frontier = [x for x in frontier if x != new_node]
                frontier += [x for x in list(graph.G.neighbors(new_node)) if x
                    not in visited]
            return graph.G.subgraph(neigh)

        pos_target = batch_target.apply_transform(sample_subgraph, offset=1)
        pos_query = pos_target.apply_transform(sample_subgraph)
        neg_target = batch_neg_target.apply_transform(sample_subgraph, offset=1)
        neg_query = batch_neg_query.apply_transform(sample_subgraph, use_precomp_sizes=False)

        return pos_target, pos_query, neg_target, neg_query

    def gen_data_loaders(self, batch_size, train=True):
        return [TorchDataLoader(self.train if train else self.test,
            collate_fn=Batch.collate([]),
            batch_size=batch_size // 2, shuffle=True) for i in range(3)]
