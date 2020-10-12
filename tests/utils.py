import numpy as np
import networkx as nx
import torch
import itertools
np.random.seed(0)


def simple_networkx_graph(directed=True):
    num_nodes = 10
    edge_index = (
        torch.tensor(
            [
                [0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 6, 6, 7, 7, 9],
                [1, 2, 2, 3, 3, 8, 4, 5, 6, 5, 6, 7, 8, 9, 8, 9, 8]
            ]
        ).long()
    )
    x = torch.zeros([num_nodes, 2])
    y = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4]).long()
    for i in range(num_nodes):
        x[i] = np.random.randint(1, num_nodes)
    edge_x = torch.zeros([edge_index.shape[1], 2])
    edge_y = torch.tensor(
        [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3]
    ).long()
    for i in range(edge_index.shape[1]):
        edge_x[i] = np.random.randint(1, num_nodes)

    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    for i, (u, v) in enumerate(edge_index.T.tolist()):
        G.add_edge(u, v)

    # if it is undirected, modify the edge attributes
    if directed is False:
        G = G.to_undirected()
        H = G.to_directed()
        edge_index = np.zeros([2, edge_index.shape[1] * 2]).astype(np.int64)
        edge_x = np.zeros([edge_x.shape[0] * 2, edge_x.shape[1]])
        edge_y = np.zeros(edge_y.shape[0] * 2).astype(np.int64)
        for i, nx_edge in enumerate(nx.to_edgelist(H)):
            edge_index[:, i] = (
                np.array([nx_edge[0], nx_edge[1]]).astype(np.int64)
            )
            edge_x[i] = nx_edge[2]['edge_attr']
            edge_y[i] = nx_edge[2]['edge_y']

    graph_x = torch.tensor([[0, 1]])
    graph_y = torch.tensor([0])
    return G, x, y, edge_x, edge_y, edge_index, graph_x, graph_y


def simple_networkx_graph_alphabet(directed=True):
    num_nodes = 10
    edge_index = (
        torch.tensor(
            [
                [0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 6, 6, 7, 7, 9],
                [1, 2, 2, 3, 3, 8, 4, 5, 6, 5, 6, 7, 8, 9, 8, 9, 8]
            ]
        ).long()
    )
    x = torch.zeros([num_nodes, 2])
    y = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4]).long()
    for i in range(num_nodes):
        x[i] = np.random.randint(1, num_nodes)
    edge_x = torch.zeros([edge_index.shape[1], 2])
    edge_y = torch.tensor(
        [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3]
    ).long()
    for i in range(edge_index.shape[1]):
        edge_x[i] = np.random.randint(1, num_nodes)

    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    for i, (u, v) in enumerate(edge_index.T.tolist()):
        G.add_edge(u, v)

    # if it is undirected, modify the edge attributes
    if directed is False:
        G = G.to_undirected()
        H = G.to_directed()
        edge_index = np.zeros([2, edge_index.shape[1] * 2]).astype(np.int64)
        edge_x = np.zeros([edge_x.shape[0] * 2, edge_x.shape[1]])
        edge_y = np.zeros(edge_y.shape[0] * 2).astype(np.int64)
        for i, nx_edge in enumerate(nx.to_edgelist(H)):
            edge_index[:, i] = (
                np.array([nx_edge[0], nx_edge[1]]).astype(np.int64)
            )
            edge_x[i] = nx_edge[2]['edge_attr']
            edge_y[i] = nx_edge[2]['edge_y']

    graph_x = torch.tensor([[0, 1]])
    graph_y = torch.tensor([0])

    # number -> alphabet transform
    keys = list(G.nodes)
    vals = [chr(x + 97) for x in list(range(len(keys)))]
    mapping = dict(zip(keys, vals))
    G = nx.relabel_nodes(G, mapping, copy=True)
    return G, x, y, edge_x, edge_y, edge_index, graph_x, graph_y


def simple_networkx_multigraph():
    num_nodes = 10
    edge_index = (
        torch.tensor(
            [
                [0, 0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 6, 6, 6, 7, 7, 9],
                [1, 1, 1, 2, 2, 3, 3, 8, 8, 4, 5, 6, 5, 6, 7, 8, 8, 9, 8, 9, 8]
            ]
        ).long()
    )
    x = torch.zeros([num_nodes, 2])
    y = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4]).long()
    for i in range(num_nodes):
        x[i] = np.random.randint(1, num_nodes)
    edge_x = torch.zeros([edge_index.shape[1], 2])
    edge_y = torch.tensor(
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3]
    ).long()
    for i in range(edge_index.shape[1]):
        edge_x[i] = np.random.randint(1, num_nodes)

    G = nx.MultiDiGraph()
    G.add_nodes_from(range(num_nodes))
    for i, (u, v) in enumerate(edge_index.T.tolist()):
        G.add_edge(u, v)

    graph_x = torch.tensor([[0, 1]])
    graph_y = torch.tensor([0])
    return G, x, y, edge_x, edge_y, edge_index, graph_x, graph_y


def sample_neigh(graph, size):
    while True:
        start_node = np.random.choice(list(graph.nodes))
        neigh = [start_node]
        frontier = list(set(graph.neighbors(start_node)) - set(neigh))
        visited = set([start_node])
        while len(neigh) < size and frontier:
            new_node = np.random.choice(list(frontier))
            assert new_node not in neigh
            neigh.append(new_node)
            visited.add(new_node)
            frontier += list(graph.neighbors(new_node))
            frontier = [x for x in frontier if x not in visited]
        if len(neigh) == size:
            return graph, neigh


def gen_graph(size, graph):
    graph, neigh = sample_neigh(graph, size)
    return graph.subgraph(neigh)


def generate_simple_hete_graph(no_edge_type=False):
    G = nx.DiGraph()
    for i in range(9):
        if i < 2:
            node_feature = torch.rand([10, ])
            node_type = "n1"
            node_label = 0
            G.add_node(
                i,
                node_type=node_type,
                node_label=node_label,
                node_feature=node_feature
            )
        elif 2 <= i < 4:
            node_feature = torch.rand([12, ])
            node_type = "n2"
            node_label = 0
            G.add_node(
                i,
                node_type=node_type,
                node_label=node_label,
                node_feature=node_feature
            )
        elif 4 <= i < 6:
            node_feature = torch.rand([10, ])
            node_type = "n1"
            node_label = 1
            G.add_node(
                i,
                node_type=node_type,
                node_label=node_label,
                node_feature=node_feature
            )
        else:
            node_feature = torch.rand([12, ])
            node_type = "n2"
            node_label = 1
            G.add_node(
                i,
                node_type=node_type,
                node_label=node_label,
                node_feature=node_feature
            )
    if no_edge_type:
        G.add_edge(0, 1, edge_label=0, edge_feature=torch.rand([8, ]))
        G.add_edge(0, 2, edge_label=1, edge_feature=torch.rand([8, ]))
        G.add_edge(0, 5, edge_label=0, edge_feature=torch.rand([8, ]))
        G.add_edge(1, 3, edge_label=0, edge_feature=torch.rand([8, ]))
        G.add_edge(1, 5, edge_label=1, edge_feature=torch.rand([8, ]))
        G.add_edge(2, 3, edge_label=1, edge_feature=torch.rand([8, ]))
        G.add_edge(2, 4, edge_label=2, edge_feature=torch.rand([8, ]))
        G.add_edge(3, 4, edge_label=2, edge_feature=torch.rand([8, ]))
        G.add_edge(4, 0, edge_label=1, edge_feature=torch.rand([8, ]))
        G.add_edge(4, 5, edge_label=1, edge_feature=torch.rand([8, ]))
        G.add_edge(5, 7, edge_label=1, edge_feature=torch.rand([8, ]))
        G.add_edge(6, 1, edge_label=1, edge_feature=torch.rand([8, ]))
        G.add_edge(6, 2, edge_label=1, edge_feature=torch.rand([8, ]))
        G.add_edge(7, 3, edge_label=2, edge_feature=torch.rand([8, ]))
        G.add_edge(8, 0, edge_label=0, edge_feature=torch.rand([8, ]))
        G.add_edge(8, 1, edge_label=0, edge_feature=torch.rand([8, ]))
    else:
        G.add_edge(
            0, 1, edge_label=0, edge_feature=torch.rand([8, ]), edge_type="e1"
        )
        G.add_edge(
            0, 2, edge_label=1, edge_feature=torch.rand([12, ]), edge_type="e2"
        )
        G.add_edge(
            0, 5, edge_label=0, edge_feature=torch.rand([8, ]), edge_type="e1"
        )
        G.add_edge(
            1, 3, edge_label=0, edge_feature=torch.rand([12, ]), edge_type="e2"
        )
        G.add_edge(
            1, 5, edge_label=1, edge_feature=torch.rand([12, ]), edge_type="e2"
        )
        G.add_edge(
            2, 3, edge_label=1, edge_feature=torch.rand([8, ]), edge_type="e1"
        )
        G.add_edge(
            2, 4, edge_label=2, edge_feature=torch.rand([12, ]), edge_type="e2"
        )
        G.add_edge(
            3, 4, edge_label=2, edge_feature=torch.rand([12, ]), edge_type="e2"
        )
        G.add_edge(
            4, 0, edge_label=1, edge_feature=torch.rand([12, ]), edge_type="e2"
        )
        G.add_edge(
            4, 5, edge_label=1, edge_feature=torch.rand([8, ]), edge_type="e1"
        )
        G.add_edge(
            5, 7, edge_label=1, edge_feature=torch.rand([8, ]), edge_type="e1"
        )
        G.add_edge(
            6, 1, edge_label=1, edge_feature=torch.rand([8, ]), edge_type="e1"
        )
        G.add_edge(
            6, 2, edge_label=1, edge_feature=torch.rand([8, ]), edge_type="e1"
        )
        G.add_edge(
            7, 3, edge_label=2, edge_feature=torch.rand([8, ]), edge_type="e1"
        )
        G.add_edge(
            8, 0, edge_label=0, edge_feature=torch.rand([12, ]), edge_type="e2"
        )
        G.add_edge(
            8, 1, edge_label=0, edge_feature=torch.rand([12, ]), edge_type="e2"
        )
    return G


def generate_simple_hete_dataset(no_edge_type=False):
    G = nx.DiGraph()
    for i in range(9):
        if i < 2:
            node_feature = torch.rand([10, ])
            node_type = "n1"
            node_label = 0
            G.add_node(
                i,
                node_type=node_type,
                node_label=node_label,
                node_feature=node_feature,
            )
        elif 2 <= i < 4:
            node_feature = torch.rand([12, ])
            node_type = "n2"
            node_label = 0
            G.add_node(
                i,
                node_type=node_type,
                node_label=node_label,
                node_feature=node_feature,
            )
        elif 4 <= i < 6:
            node_feature = torch.rand([10, ])
            node_type = "n1"
            node_label = 1
            G.add_node(
                i,
                node_type=node_type,
                node_label=node_label,
                node_feature=node_feature,
            )
        else:
            node_feature = torch.rand([12, ])
            node_type = "n2"
            node_label = 1
            G.add_node(
                i,
                node_type=node_type,
                node_label=node_label,
                node_feature=node_feature,
            )
    if no_edge_type:
        G.add_edge(0, 1, edge_feature=torch.rand([8, ]))
        G.add_edge(0, 2, edge_feature=torch.rand([8, ]))
        G.add_edge(0, 5, edge_feature=torch.rand([8, ]))
        G.add_edge(1, 3, edge_feature=torch.rand([8, ]))
        G.add_edge(1, 5, edge_feature=torch.rand([8, ]))
        G.add_edge(2, 3, edge_feature=torch.rand([8, ]))
        G.add_edge(2, 4, edge_feature=torch.rand([8, ]))
        G.add_edge(3, 4, edge_feature=torch.rand([8, ]))
        G.add_edge(4, 0, edge_feature=torch.rand([8, ]))
        G.add_edge(4, 5, edge_feature=torch.rand([8, ]))
        G.add_edge(5, 7, edge_feature=torch.rand([8, ]))
        G.add_edge(6, 1, edge_feature=torch.rand([8, ]))
        G.add_edge(6, 2, edge_feature=torch.rand([8, ]))
        G.add_edge(7, 3, edge_feature=torch.rand([8, ]))
        G.add_edge(8, 0, edge_feature=torch.rand([8, ]))
        G.add_edge(8, 1, edge_feature=torch.rand([8, ]))
    else:
        G.add_edge(0, 1, edge_feature=torch.rand([8, ]), edge_type="e1")
        G.add_edge(0, 2, edge_feature=torch.rand([12, ]), edge_type="e2")
        G.add_edge(0, 5, edge_feature=torch.rand([8, ]), edge_type="e1")
        G.add_edge(1, 3, edge_feature=torch.rand([12, ]), edge_type="e2")
        G.add_edge(1, 5, edge_feature=torch.rand([12, ]), edge_type="e2")
        G.add_edge(2, 3, edge_feature=torch.rand([8, ]), edge_type="e1")
        G.add_edge(2, 4, edge_feature=torch.rand([12, ]), edge_type="e2")
        G.add_edge(3, 4, edge_feature=torch.rand([12, ]), edge_type="e2")
        G.add_edge(4, 0, edge_feature=torch.rand([12, ]), edge_type="e2")
        G.add_edge(4, 5, edge_feature=torch.rand([8, ]), edge_type="e1")
        G.add_edge(5, 7, edge_feature=torch.rand([8, ]), edge_type="e1")
        G.add_edge(6, 1, edge_feature=torch.rand([8, ]), edge_type="e1")
        G.add_edge(6, 2, edge_feature=torch.rand([8, ]), edge_type="e1")
        G.add_edge(7, 3, edge_feature=torch.rand([8, ]), edge_type="e1")
        G.add_edge(8, 0, edge_feature=torch.rand([12, ]), edge_type="e2")
        G.add_edge(8, 1, edge_feature=torch.rand([12, ]), edge_type="e2")
    return G


def generate_dense_hete_graph(no_edge_type=False):
    G = nx.DiGraph()
    num_node = 20
    for i in range(num_node):
        if i < 10:
            node_feature = torch.rand([10, ])
            node_type = "n1"
            node_label = 0
            G.add_node(
                i,
                node_type=node_type,
                node_label=node_label,
                node_feature=node_feature,
            )
        else:
            node_feature = torch.rand([12, ])
            node_type = "n2"
            node_label = 1
            G.add_node(
                i,
                node_type=node_type,
                node_label=node_label,
                node_feature=node_feature,
            )

    if no_edge_type:
        for i, j in itertools.permutations(range(num_node), 2):
            rand = np.random.random()
            if (rand > 0.8):
                continue
            elif rand > 0.4:
                G.add_edge(i, j, edge_label=0, edge_feature=torch.rand([8, ]))
            else:
                G.add_edge(i, j, edge_label=0, edge_feature=torch.rand([8, ]))
    else:
        for i, j in itertools.permutations(range(num_node), 2):
            rand = np.random.random()
            if (rand > 0.8):
                continue
            elif rand > 0.4:
                G.add_edge(
                    i,
                    j,
                    edge_label=0,
                    edge_feature=torch.rand([8, ]),
                    edge_type='e1',
                )
            else:
                G.add_edge(
                    i,
                    j,
                    edge_label=0,
                    edge_feature=torch.rand([8, ]),
                    edge_type='e2',
                )
    return G


def generate_dense_hete_dataset(no_edge_type=False):
    G = nx.DiGraph()
    num_node = 20
    for i in range(num_node):
        if i < 10:
            node_feature = torch.rand([1, ])
            node_type = "n1"
            node_label = 0
            G.add_node(
                i,
                node_type=node_type,
                node_label=node_label,
                node_feature=node_feature,
            )
        else:
            node_feature = torch.rand([1, ])
            node_type = "n2"
            node_label = 1
            G.add_node(
                i,
                node_type=node_type,
                node_label=node_label,
                node_feature=node_feature,
            )

    if no_edge_type:
        for i, j in itertools.permutations(range(num_node), 2):
            rand = np.random.random()
            if (rand > 0.8):
                continue
            elif rand > 0.4:
                G.add_edge(i, j, edge_feature=torch.rand([1, ]))
            else:
                G.add_edge(i, j, edge_feature=torch.rand([1, ]))
    else:
        for i, j in itertools.permutations(range(num_node), 2):
            rand = np.random.random()
            if (rand > 0.8):
                continue
            elif rand > 0.4:
                G.add_edge(
                    i,
                    j,
                    edge_feature=torch.rand([1, ]),
                    edge_type='e1',
                )
            else:
                G.add_edge(
                    i,
                    j,
                    edge_feature=torch.rand([1, ]),
                    edge_type='e2',
                )
    return G


def generate_dense_hete_multigraph(no_edge_type=False):
    G = nx.MultiDiGraph()
    num_node = 20
    for i in range(num_node):
        if i < 10:
            node_feature = torch.rand([10, ])
            node_type = "n1"
            node_label = 0
            G.add_node(
                i,
                node_type=node_type,
                node_label=node_label,
                node_feature=node_feature,
            )
        else:
            node_feature = torch.rand([12, ])
            node_type = "n2"
            node_label = 1
            G.add_node(
                i,
                node_type=node_type,
                node_label=node_label,
                node_feature=node_feature,
            )

    if no_edge_type:
        for i, j in itertools.permutations(range(num_node), 2):
            rand = np.random.random()
            if (rand > 0.8):
                continue
            elif rand > 0.4:
                G.add_edge(i, j, edge_label=0, edge_feature=torch.rand([8, ]))
                G.add_edge(i, j, edge_label=0, edge_feature=torch.rand([8, ]))
            else:
                G.add_edge(i, j, edge_label=0, edge_feature=torch.rand([8, ]))
                G.add_edge(i, j, edge_label=0, edge_feature=torch.rand([8, ]))
    else:
        for i, j in itertools.permutations(range(num_node), 2):
            rand = np.random.random()
            if (rand > 0.8):
                continue
            elif rand > 0.4:
                G.add_edge(
                    i,
                    j,
                    edge_label=0,
                    edge_feature=torch.rand([8, ]),
                    edge_type='e1',
                )
                G.add_edge(
                    i,
                    j,
                    edge_label=0,
                    edge_feature=torch.rand([8, ]),
                    edge_type='e1',
                )
            else:
                G.add_edge(
                    i,
                    j,
                    edge_label=0,
                    edge_feature=torch.rand([8, ]),
                    edge_type='e2',
                )
                G.add_edge(
                    i,
                    j,
                    edge_label=0,
                    edge_feature=torch.rand([8, ]),
                    edge_type='e2',
                )
    return G
