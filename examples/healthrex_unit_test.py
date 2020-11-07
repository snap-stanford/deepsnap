import pandas as pd
import numpy as np
import networkx as nx
from deepsnap.graph import Graph
from deepsnap.dataset import GraphDataset
from deepsnap.hetero_graph import HeteroGraph
import torch
# import snap


def construct_graph():
    # Create the nx graph
    G = nx.DiGraph()
    edges_train, edges_train_disjoint, edges_val, edges_test = [], [], [], []

    # Add the s nodes
    n_s = 30
    for idx in range(n_s):
        G.add_node('s{}'.format(idx), node_type='s', node_feature=torch.zeros(110, dtype=torch.float))

    # Add the o nodes
    n_o = 60
    for idx in range(n_o):
        G.add_node('o{}'.format(idx), node_type='o', node_feature=torch.zeros(110, dtype=torch.float))

    # Add all the p nodes, features, and edges
    n_p = 5500
    for index in range(n_p):
        # Extract features
        features_1 = torch.rand(60, dtype=torch.float)
        features_2 = torch.rand(100, dtype=torch.float)

        # Add p nodes
        G.add_node('p{}'.format(index), node_type='p', node_feature=torch.cat((features_1, features_2)))

        # Add s-p edges
        n_s = 30
        for idx in range(n_s):
            rand = np.random.random()
            if (rand > 0.5):
                G.add_edge('p{}'.format(index), 's{}'.format(idx), edge_type='s-p', edge_label=0)
                G.add_edge('s{}'.format(idx), 'p{}'.format(index), edge_type='s-p', edge_label=0)

                e1 = ('s{}'.format(idx), 'p{}'.format(index),  {'edge_type': 's-p', 'edge_label': 0})
                e2 = ('p{}'.format(index), 's{}'.format(idx),  {'edge_type': 's-p', 'edge_label':0 })
                edges_train += [e1, e2]
                edges_val += [e1, e2]
                edges_test += [e1, e2]

        # Add target edges of type pr
        n_t = 60
        for idx in range(n_t):
            rand = np.random.random()
            if (rand > 0.5):
                e1 = ('p{}'.format(index), 'o{}'.format(idx), {'edge_type': 'pr', 'edge_label': 1})
                e2 = ('o{}'.format(idx), 'p{}'.format(index), {'edge_type': 'pr', 'edge_label': 1})

                G.add_edge('p{}'.format(index), 'o{}'.format(idx), edge_type='pr', edge_label=1)
                G.add_edge('o{}'.format(idx), 'p{}'.format(index), edge_type='pr', edge_label=1)

                if idx >= 0 and idx < n_o*0.8:
                    edges_train += [e1, e2]
                    edges_train_disjoint += [e1, e2]
                elif idx >= n_o*0.8 and idx < n_o*0.9:
                    edges_val += [e1, e2]
                else:
                    edges_test += [e1, e2]

        # Add non-target edges of type pr
        n_o = 60
        for idx in range(n_o):
            rand = np.random.random()
            if (rand > 0.5):
                e1 = ('p{}'.format(index), 'o{}'.format(idx), {'edge_type': 'pr', 'edge_label': 0})
                e2 = ('o{}'.format(idx), 'p{}'.format(index), {'edge_type': 'pr', 'edge_label': 0})

                for e in [e1, e2]:
                    # Check for duplicates
                    if not G.has_edge(e[0], e[1]):
                        G.add_edge(e[0], e[1], edge_type=e[2]['edge_type'], edge_label=0)
                        edges_train += [e]
                        edges_val += [e]
                        edges_test += [e]


    return G, edges_train, edges_val, edges_test, edges_train_disjoint



G, edges_train, edges_val, edges_test, edges_train_disjoint = construct_graph()
link_split_types = [("o", "pr", "p"), ("p", "pr", "o")]
split_dim = [0.8, 0.1, 0.1]

hete = HeteroGraph(
    G,
    custom={
        "general_splits": [
            edges_train,
            edges_val,
            edges_test
        ],
        "disjoint_split": edges_train_disjoint,
        "task": "link_pred",
    }
)


dataset = GraphDataset(
    [hete],
    task="link_pred",
    edge_train_mode="disjoint"
)


dataset_train, dataset_val, dataset_test = dataset.split(
    transductive=True,
    split_types=link_split_types
)

# Should be 60
edge_label_index_split_type_length = dataset_train[0].edge_label_index[('p', 'pr', 'o')].shape[1]
print(dataset_train[0].edge_label_index[('p', 'pr', 'o')][0].max())
print(dataset_train[0].edge_label_index[('p', 'pr', 'o')][1].max())
print(dataset_train[0].edge_label_index[('o', 'pr', 'p')][0].max())
print(dataset_train[0].edge_label_index[('o', 'pr', 'p')][1].max())

