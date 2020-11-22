import random
import torch
import unittest
from torch_geometric.datasets import TUDataset, Planetoid
import copy
from copy import deepcopy
from deepsnap.graph import Graph
from deepsnap.hetero_graph import HeteroGraph
from deepsnap.dataset import GraphDataset, Generator, EnsembleGenerator
from tests.utils import (
    pyg_to_dicts,
    simple_networkx_graph,
    simple_networkx_graph_alphabet,
    simple_networkx_multigraph,
    generate_dense_hete_dataset,
    gen_graph,
)


class TestDatasetTensor(unittest.TestCase):

    def test_dataset_basic(self):
        _, x, y, edge_x, edge_y, edge_index, graph_x, graph_y = (
            simple_networkx_graph()
        )

        G = Graph(
            node_feature=x, node_label=y, edge_index=edge_index,
            edge_feature=edge_x, edge_label=edge_y,
            graph_feature=graph_x, graph_label=graph_y, directed=True
        )

        H = deepcopy(G)

        dataset = GraphDataset([G, H])
        self.assertEqual(len(dataset), 2)

    def test_dataset_property(self):
        _, x, y, edge_x, edge_y, edge_index, graph_x, graph_y = (
            simple_networkx_graph()
        )
        G = Graph(
            node_feature=x, node_label=y, edge_index=edge_index,
            edge_feature=edge_x, edge_label=edge_y,
            graph_feature=graph_x, graph_label=graph_y, directed=True
        )

        H = deepcopy(G)

        H.graph_label = torch.tensor([1])

        graphs = [G, H]
        dataset = GraphDataset(graphs)
        self.assertEqual(dataset.num_node_labels, 5)
        self.assertEqual(dataset.num_node_features, 2)
        self.assertEqual(dataset.num_edge_labels, 4)
        self.assertEqual(dataset.num_edge_features, 2)
        self.assertEqual(dataset.num_graph_labels, 2)
        self.assertEqual(dataset.num_graph_features, 2)
        self.assertEqual(dataset.num_labels, 5)  # node task
        dataset = GraphDataset(graphs, task="edge")
        self.assertEqual(dataset.num_labels, 4)
        dataset = GraphDataset(graphs, task="link_pred")
        self.assertEqual(dataset.num_labels, 4)
        dataset = GraphDataset(graphs, task="graph")
        self.assertEqual(dataset.num_labels, 2)

    def test_dataset_split(self):
        # inductively split with graph task
        pyg_dataset = TUDataset("./enzymes", "ENZYMES")
        ds = pyg_to_dicts(pyg_dataset)
        graphs = [Graph(**item) for item in ds]
        dataset = GraphDataset(graphs, task="graph")
        split_res = dataset.split(transductive=False)
        num_graphs = len(dataset)
        num_graphs_reduced = num_graphs - 3
        num_train = 1 + int(num_graphs_reduced * 0.8)
        num_val = 1 + int(num_graphs_reduced * 0.1)
        num_test = num_graphs - num_train - num_val
        self.assertEqual(num_train, len(split_res[0]))
        self.assertEqual(num_val, len(split_res[1]))
        self.assertEqual(num_test, len(split_res[2]))

        # inductively split with link_pred task
        # and default (`all`) edge_train_mode
        pyg_dataset = TUDataset("./enzymes", "ENZYMES")
        ds = pyg_to_dicts(pyg_dataset)
        graphs = [Graph(**item) for item in ds]
        dataset = GraphDataset(graphs, task="link_pred")
        split_res = dataset.split(transductive=False)
        num_graphs = len(dataset)
        num_graphs_reduced = num_graphs - 3
        num_train = 1 + int(num_graphs_reduced * 0.8)
        num_val = 1 + int(num_graphs_reduced * 0.1)
        num_test = num_graphs - num_train - num_val
        self.assertEqual(num_train, len(split_res[0]))
        self.assertEqual(num_val, len(split_res[1]))
        self.assertEqual(num_test, len(split_res[2]))

        # inductively split with link_pred task and `disjoint` edge_train_mode
        pyg_dataset = TUDataset("./enzymes", "ENZYMES")
        ds = pyg_to_dicts(pyg_dataset)
        graphs = [Graph(**item) for item in ds]
        dataset = GraphDataset(
            graphs,
            task="link_pred",
            edge_train_mode="disjoint",
        )
        split_res = dataset.split(transductive=False)
        num_graphs = len(dataset)
        num_graphs_reduced = num_graphs - 3
        num_train = 1 + int(num_graphs_reduced * 0.8)
        num_val = 1 + int(num_graphs_reduced * 0.1)
        num_test = num_graphs - num_train - num_val
        self.assertEqual(num_train, len(split_res[0]))
        self.assertEqual(num_val, len(split_res[1]))
        self.assertEqual(num_test, len(split_res[2]))

        # transductively split with node task
        pyg_dataset = Planetoid("./cora", "Cora")
        ds = pyg_to_dicts(pyg_dataset, task="cora")
        graphs = [Graph(**item) for item in ds]
        dataset = GraphDataset(graphs, task="node")
        num_nodes = dataset.num_nodes[0]
        num_nodes_reduced = num_nodes - 3
        num_edges = dataset.num_edges[0]
        num_edges_reduced = num_edges - 3
        split_res = dataset.split()
        self.assertEqual(
            len(split_res[0][0].node_label_index),
            1 + int(0.8 * num_nodes_reduced)
        )
        self.assertEqual(
            len(split_res[1][0].node_label_index),
            1 + int(0.1 * num_nodes_reduced)
        )
        self.assertEqual(
            len(split_res[2][0].node_label_index),
            num_nodes
            - 2
            - int(0.8 * num_nodes_reduced)
            - int(0.1 * num_nodes_reduced)
        )

        # transductively split with link_pred task
        # and default (`all`) edge_train_mode
        dataset = GraphDataset(graphs, task="link_pred")
        split_res = dataset.split()
        self.assertEqual(
            split_res[0][0].edge_label_index.shape[1],
            2 * 2 * (1 + int(0.8 * (num_edges_reduced)))
        )
        self.assertEqual(
            split_res[1][0].edge_label_index.shape[1],
            2
            * 2 * (1 + (int(0.1 * (num_edges_reduced))))
        )
        self.assertEqual(
            split_res[2][0].edge_label_index.shape[1],
            2
            * 2
            * num_edges
            - 2
            * 2
            * (
                2
                + int(0.1 * num_edges_reduced)
                + int(0.8 * num_edges_reduced)
            )
        )

        # transductively split with link_pred task, `split` edge_train_mode
        # and 0.5 edge_message_ratio
        dataset = GraphDataset(
            graphs,
            task="link_pred",
            edge_train_mode="disjoint",
            edge_message_ratio=0.5,
        )
        split_res = dataset.split()
        edge_0 = 2 * (1 + int(0.8 * num_edges_reduced))
        edge_0 = 2 * (edge_0 - (1 + int(0.5 * (edge_0 - 3))))
        self.assertEqual(
            split_res[0][0].edge_label_index.shape[1],
            edge_0,
        )
        edge_1 = 2 * 2 * (1 + int(0.1 * num_edges_reduced))
        self.assertEqual(split_res[1][0].edge_label_index.shape[1], edge_1)
        edge_2 = (
            2
            * 2
            * int(num_edges)
            - 2
            * 2 * (1 + int(0.8 * num_edges_reduced))
            - edge_1
        )

        self.assertEqual(split_res[2][0].edge_label_index.shape[1], edge_2)

        # transductively split with link_pred task
        # and specified edge_negative_sampling_ratio
        dataset = GraphDataset(
            graphs,
            task="link_pred",
            edge_negative_sampling_ratio=2
        )
        split_res = dataset.split()
        edge_0 = (2 + 1) * (2 * (1 + int(0.8 * num_edges_reduced)))
        self.assertEqual(split_res[0][0].edge_label_index.shape[1], edge_0)
        edge_1 = (2 + 1) * 2 * (1 + int(0.1 * num_edges_reduced))
        self.assertEqual(split_res[1][0].edge_label_index.shape[1], edge_1)
        edge_2 = (2 + 1) * 2 * int(num_edges) - edge_0 - edge_1
        self.assertEqual(split_res[2][0].edge_label_index.shape[1], edge_2)

    def test_dataset_split_custom(self):
        # transductive split with node task (self defined dataset)
        G, x, y, edge_x, edge_y, edge_index, graph_x, graph_y = (
            simple_networkx_graph()
        )
        Graph.add_edge_attr(G, "edge_feature", edge_x)
        Graph.add_edge_attr(G, "edge_label", edge_y)
        Graph.add_node_attr(G, "node_feature", x)
        Graph.add_node_attr(G, "node_label", y)
        Graph.add_graph_attr(G, "graph_feature", graph_x)
        Graph.add_graph_attr(G, "graph_label", graph_y)

        num_nodes = len(list(G.nodes))
        nodes_train = list(G.nodes)[: int(0.3 * num_nodes)]
        nodes_val = list(G.nodes)[int(0.3 * num_nodes): int(0.6 * num_nodes)]
        nodes_test = list(G.nodes)[int(0.6 * num_nodes):]

        graph = Graph(
            node_feature=x, node_label=y, edge_index=edge_index,
            edge_feature=edge_x, edge_label=edge_y,
            graph_feature=graph_x, graph_label=graph_y, directed=True,
            custom={
                "general_splits": [
                    nodes_train,
                    nodes_val,
                    nodes_test
                ],
                "task": "node"
            }
        )

        graphs = [graph]
        dataset = GraphDataset(
            graphs, task="node"
        )

        split_res = dataset.split(transductive=True)
        self.assertEqual(
            split_res[0][0].node_label_index,
            list(range(int(0.3 * num_nodes)))
        )
        self.assertEqual(
            split_res[1][0].node_label_index,
            list(range(int(0.3 * num_nodes), int(0.6 * num_nodes)))
        )
        self.assertEqual(
            split_res[2][0].node_label_index,
            list(range(int(0.6 * num_nodes), num_nodes))
        )

        # transductive split with link_pred task (train/val split)
        # edges = list(G.edges)
        # num_edges = len(edges)
        # edges_train = edges[: int(0.7 * num_edges)]
        # edges_val = edges[int(0.7 * num_edges):]
        # link_size_list = [len(edges_train), len(edges_val)]

        # graph = Graph(
        #     node_feature=x, node_label=y, edge_index=edge_index,
        #     edge_feature=edge_x, edge_label=edge_y,
        #     graph_feature=graph_x, graph_label=graph_y, directed=True,
        #     custom={
        #         "general_splits": [
        #             edges_train,
        #             edges_val
        #         ],
        #         "task": "link_pred"
        #     }
        # )

        # graphs = [graph]
        # dataset = GraphDataset(
        #     graphs,
        #     task="link_pred"
        # )

        # split_res = dataset.split(transductive=True)

        # self.assertEqual(
        #     split_res[0][0].edge_label_index.shape[1],
        #     2 * link_size_list[0]
        # )
        # self.assertEqual(
        #     split_res[1][0].edge_label_index.shape[1],
        #     2 * link_size_list[1]
        # )

    #     # transductive split with link_pred task (custom negative sampling) (larger/equal amount) (train/val split)
    #     edges = list(G.edges)
    #     num_edges = len(edges)
    #     edges_train = edges[: int(0.7 * num_edges)]
    #     edges_val = edges[int(0.7 * num_edges):]
    #     custom_negative_sampling_train = [
    #         ("a", "a") for _ in range(len(edges_train))
    #     ]
    #     custom_negative_sampling_val = [
    #         ("b", "b") for _ in range(len(edges_val))
    #     ]
    #     link_size_list = [len(edges_train), len(edges_val)]

    #     graph = Graph(
    #         G,
    #         custom={
    #             "general_splits": [
    #                 edges_train,
    #                 edges_val
    #             ],
    #             "negative_edges": [
    #                 custom_negative_sampling_train,
    #                 custom_negative_sampling_val
    #             ],
    #             "task": "link_pred"
    #         }
    #     )

    #     graphs = [graph]
    #     dataset = GraphDataset(
    #         graphs,
    #         task="link_pred"
    #     )

    #     split_res = dataset.split(transductive=True)

    #     self.assertEqual(
    #         split_res[0][0].edge_label_index.shape[1],
    #         2 * link_size_list[0]
    #     )
    #     self.assertEqual(
    #         split_res[1][0].edge_label_index.shape[1],
    #         2 * link_size_list[1]
    #     )
    #     self.assertEqual(
    #         split_res[0][0].edge_label_index[:, len(edges_train):].tolist(),
    #         [list(x) for x in list(zip(*custom_negative_sampling_train))]
    #     )
    #     self.assertEqual(
    #         split_res[1][0].edge_label_index[:, len(edges_val):].tolist(),
    #         [list(x) for x in list(zip(*custom_negative_sampling_val))]
    #     )

    #     # transductive split with link_pred task (custom negative sampling) (smaller amount) (train/val split)
    #     edges = list(G.edges)
    #     num_edges = len(edges)
    #     edges_train = edges[: int(0.7 * num_edges)]
    #     edges_val = edges[int(0.7 * num_edges):]
    #     custom_negative_sampling_train = [("a", "a")]
    #     custom_negative_sampling_val = [("b", "b")]
    #     link_size_list = [len(edges_train), len(edges_val)]

    #     graph = Graph(
    #         G,
    #         custom={
    #             "general_splits": [
    #                 edges_train,
    #                 edges_val
    #             ],
    #             "negative_edges": [
    #                 custom_negative_sampling_train,
    #                 custom_negative_sampling_val
    #             ],
    #             "task": "link_pred"
    #         }
    #     )

    #     graphs = [graph]
    #     dataset = GraphDataset(
    #         graphs,
    #         task="link_pred"
    #     )

    #     split_res = dataset.split(transductive=True)

    #     self.assertEqual(
    #         split_res[0][0].edge_label_index.shape[1],
    #         2 * link_size_list[0]
    #     )
    #     self.assertEqual(
    #         split_res[1][0].edge_label_index.shape[1],
    #         2 * link_size_list[1]
    #     )
    #     self.assertEqual(
    #         split_res[0][0].edge_label_index[:, len(edges_train):].tolist(),
    #         [
    #             len(edges_train) * list(x)
    #             for x in list(zip(*custom_negative_sampling_train))
    #         ]
    #     )
    #     self.assertEqual(
    #         split_res[1][0].edge_label_index[:, len(edges_val):].tolist(),
    #         [
    #             len(edges_val) * list(x)
    #             for x in list(zip(*custom_negative_sampling_val))
    #         ]
    #     )

    #     # transductive split with link_pred task (disjoint mode) (self defined dataset) (train/val/test split)
    #     edges = list(G.edges)
    #     num_edges = len(edges)
    #     edges_train = edges[: int(0.3 * num_edges)]
    #     edges_train_disjoint = edges[: int(0.5 * 0.3 * num_edges)]
    #     edges_val = edges[int(0.3 * num_edges): int(0.6 * num_edges)]
    #     edges_test = edges[int(0.6 * num_edges):]
    #     link_size_list = [
    #         len(edges_train_disjoint), len(edges_val), len(edges_test)
    #     ]

    #     graph = Graph(
    #         G,
    #         custom={
    #             "general_splits": [
    #                 edges_train,
    #                 edges_val,
    #                 edges_test
    #             ],
    #             "disjoint_split": edges_train_disjoint,
    #             "task": "link_pred"
    #         }
    #     )

    #     graphs = [graph]
    #     dataset = GraphDataset(
    #         graphs,
    #         task="link_pred",
    #         edge_train_mode="disjoint"
    #     )

    #     split_res = dataset.split(transductive=True)
    #     self.assertEqual(
    #         split_res[0][0].edge_label_index.shape[1],
    #         2 * link_size_list[0]
    #     )
    #     self.assertEqual(
    #         split_res[1][0].edge_label_index.shape[1],
    #         2 * link_size_list[1]
    #     )
    #     self.assertEqual(
    #         split_res[2][0].edge_label_index.shape[1],
    #         2 * link_size_list[2]
    #     )

    #     # transductive split with link_pred task (disjoint mode) (self defined disjoint data) (train/val split)
    #     edges = list(G.edges)
    #     num_edges = len(edges)
    #     edges_train = edges[: int(0.7 * num_edges)]
    #     edges_train_disjoint = edges[: int(0.5 * 0.7 * num_edges)]
    #     edges_val = edges[int(0.7 * num_edges):]
    #     link_size_list = [len(edges_train_disjoint), len(edges_val)]

    #     graph = Graph(
    #         G,
    #         custom={
    #             "general_splits": [
    #                 edges_train,
    #                 edges_val
    #             ],
    #             "disjoint_split": edges_train_disjoint,
    #             "task": "link_pred"
    #         }
    #     )

    #     graphs = [graph]
    #     dataset = GraphDataset(
    #         graphs,
    #         task="link_pred",
    #         edge_train_mode="disjoint"
    #     )

    #     split_res = dataset.split(transductive=True)

    #     self.assertEqual(
    #         split_res[0][0].edge_label_index.shape[1],
    #         2 * link_size_list[0]
    #     )
    #     self.assertEqual(
    #         split_res[1][0].edge_label_index.shape[1],
    #         2 * link_size_list[1]
    #     )

    #     # transductive split with link_pred task (disjoint mode) (self defined disjoint data) (multigraph) (train/val split)
    #     G, x, y, edge_x, edge_y, edge_index, graph_x, graph_y = (
    #         simple_networkx_multigraph()
    #     )
    #     Graph.add_edge_attr(G, "edge_feature", edge_x)
    #     Graph.add_edge_attr(G, "edge_label", edge_y)
    #     Graph.add_node_attr(G, "node_feature", x)
    #     Graph.add_node_attr(G, "node_label", y)
    #     Graph.add_graph_attr(G, "graph_feature", graph_x)
    #     Graph.add_graph_attr(G, "graph_label", graph_y)
    #     edges = list(G.edges)
    #     num_edges = len(edges)
    #     edges_train = edges[: int(0.6 * num_edges)]
    #     edges_train_disjoint = edges[: int(0.6 * 0.2 * num_edges)]
    #     edges_val = edges[int(0.6 * num_edges):]
    #     link_size_list = [len(edges_train_disjoint), len(edges_val)]

    #     graph = Graph(
    #         G,
    #         custom={
    #             "general_splits": [
    #                 edges_train,
    #                 edges_val
    #             ],
    #             "disjoint_split": edges_train_disjoint,
    #             "task": "link_pred"
    #         }
    #     )

    #     graphs = [graph]
    #     dataset = GraphDataset(
    #         graphs,
    #         task="link_pred",
    #         edge_train_mode="disjoint"
    #     )

    #     split_res = dataset.split(transductive=True)

    #     self.assertEqual(
    #         split_res[0][0].edge_label_index.shape[1],
    #         2 * link_size_list[0]
    #     )
    #     self.assertEqual(
    #         split_res[1][0].edge_label_index.shape[1],
    #         2 * link_size_list[1]
    #     )

    #     # transductive split with link_pred task (disjoint mode) (self defined disjoint data) (multigraph) (train/val/test split)
    #     G, x, y, edge_x, edge_y, edge_index, graph_x, graph_y = (
    #         simple_networkx_multigraph()
    #     )
    #     Graph.add_edge_attr(G, "edge_feature", edge_x)
    #     Graph.add_edge_attr(G, "edge_label", edge_y)
    #     Graph.add_node_attr(G, "node_feature", x)
    #     Graph.add_node_attr(G, "node_label", y)
    #     Graph.add_graph_attr(G, "graph_feature", graph_x)
    #     Graph.add_graph_attr(G, "graph_label", graph_y)

    #     edges = list(G.edges)
    #     num_edges = len(edges)
    #     edges_train = edges[: int(0.6 * num_edges)]
    #     edges_train_disjoint = edges[: int(0.6 * 0.2 * num_edges)]
    #     edges_val = edges[int(0.6 * num_edges):int(0.8 * num_edges)]
    #     edges_test = edges[int(0.8 * num_edges):]
    #     link_size_list = [
    #         len(edges_train_disjoint), len(edges_val), len(edges_test)
    #     ]

    #     graph = Graph(
    #         G,
    #         custom={
    #             "general_splits": [
    #                 edges_train,
    #                 edges_val,
    #                 edges_test
    #             ],
    #             "disjoint_split": edges_train_disjoint,
    #             "task": "link_pred"
    #         }
    #     )

    #     graphs = [graph]
    #     dataset = GraphDataset(
    #         graphs,
    #         task="link_pred",
    #         edge_train_mode="disjoint"
    #     )

    #     split_res = dataset.split(transductive=True)

    #     self.assertEqual(
    #         split_res[0][0].edge_label_index.shape[1],
    #         2 * link_size_list[0]
    #     )
    #     self.assertEqual(
    #         split_res[1][0].edge_label_index.shape[1],
    #         2 * link_size_list[1]
    #     )
    #     self.assertEqual(
    #         split_res[2][0].edge_label_index.shape[1],
    #         2 * link_size_list[2]
    #     )

        # transductive split with node task (pytorch geometric dataset)
        pyg_dataset = Planetoid("./cora", "Cora")
        ds = pyg_to_dicts(pyg_dataset, task="cora")
        graphs = [Graph(**item) for item in ds]
        split_ratio = [0.3, 0.3, 0.4]
        node_size_list = [0 for i in range(len(split_ratio))]
        for graph in graphs:
            custom_splits = [[] for i in range(len(split_ratio))]
            split_offset = 0
            shuffled_node_indices = torch.randperm(graph.num_nodes)
            for i, split_ratio_i in enumerate(split_ratio):
                if i != len(split_ratio) - 1:
                    num_split_i = (
                        1 +
                        int(
                            split_ratio_i *
                            (graph.num_nodes - len(split_ratio))
                        )
                    )
                    nodes_split_i = (
                        shuffled_node_indices[
                            split_offset: split_offset + num_split_i
                        ]
                    )
                    split_offset += num_split_i
                else:
                    nodes_split_i = shuffled_node_indices[split_offset:]

                custom_splits[i] = nodes_split_i
                node_size_list[i] += len(nodes_split_i)
            graph.custom = {
                "general_splits": custom_splits
            }

        dataset = GraphDataset(
            graphs, task="node"
        )

        split_res = dataset.split(transductive=True)
        self.assertEqual(
            len(split_res[0][0].node_label_index),
            node_size_list[0]
        )
        self.assertEqual(
            len(split_res[1][0].node_label_index),
            node_size_list[1]
        )
        self.assertEqual(
            len(split_res[2][0].node_label_index),
            node_size_list[2]
        )

    #     # transductive split with edge task
    #     pyg_dataset = Planetoid("./cora", "Cora")
    #     graphs = GraphDataset.pyg_to_graphs(pyg_dataset)
    #     split_ratio = [0.3, 0.3, 0.4]
    #     edge_size_list = [0 for i in range(len(split_ratio))]
    #     for graph in graphs:
    #         custom_splits = [[] for i in range(len(split_ratio))]
    #         split_offset = 0
    #         edges = list(graph.G.edges)
    #         random.shuffle(edges)
    #         for i, split_ratio_i in enumerate(split_ratio):
    #             if i != len(split_ratio) - 1:
    #                 num_split_i = (
    #                     1 +
    #                     int(
    #                         split_ratio_i
    #                         * (graph.num_edges - len(split_ratio))
    #                     )
    #                 )
    #                 edges_split_i = (
    #                     edges[split_offset: split_offset + num_split_i]
    #                 )
    #                 split_offset += num_split_i
    #             else:
    #                 edges_split_i = edges[split_offset:]

    #             custom_splits[i] = edges_split_i
    #             edge_size_list[i] += len(edges_split_i)
    #         graph.custom = {
    #             "general_splits": custom_splits
    #         }

    #     dataset = GraphDataset(
    #         graphs, task="edge"
    #     )
    #     split_res = dataset.split(transductive=True)
    #     self.assertEqual(
    #         split_res[0][0].edge_label_index.shape[1],
    #         2 * edge_size_list[0]
    #     )
    #     self.assertEqual(
    #         split_res[1][0].edge_label_index.shape[1],
    #         2 * edge_size_list[1]
    #     )
    #     self.assertEqual(
    #         split_res[2][0].edge_label_index.shape[1],
    #         2 * edge_size_list[2]
    #     )

    #     # transductive split with link_pred task
    #     pyg_dataset = Planetoid("./cora", "Cora")
    #     graphs = GraphDataset.pyg_to_graphs(pyg_dataset)
    #     split_ratio = [0.3, 0.3, 0.4]
    #     link_size_list = [0 for i in range(len(split_ratio))]

    #     for graph in graphs:
    #         split_offset = 0
    #         edges = list(graph.G.edges)
    #         random.shuffle(edges)
    #         num_edges_train = 1 + int(split_ratio[0] * (graph.num_edges - 3))
    #         num_edges_val = 1 + int(split_ratio[0] * (graph.num_edges - 3))
    #         edges_train = edges[:num_edges_train]
    #         edges_val = edges[num_edges_train:num_edges_train + num_edges_val]
    #         edges_test = edges[num_edges_train + num_edges_val:]

    #         custom_splits = [
    #             edges_train,
    #             edges_val,
    #             edges_test,
    #         ]
    #         graph.custom = {
    #             "general_splits": custom_splits
    #         }

    #         link_size_list[0] += len(edges_train)
    #         link_size_list[1] += len(edges_val)
    #         link_size_list[2] += len(edges_test)

    #     dataset = GraphDataset(
    #         graphs, task="link_pred"
    #     )
    #     split_res = dataset.split(transductive=True)
    #     self.assertEqual(
    #         split_res[0][0].edge_label_index.shape[1],
    #         2 * 2 * link_size_list[0]
    #     )
    #     self.assertEqual(
    #         split_res[1][0].edge_label_index.shape[1],
    #         2 * 2 * link_size_list[1]
    #     )
    #     self.assertEqual(
    #         split_res[2][0].edge_label_index.shape[1],
    #         2 * 2 * link_size_list[2]
    #     )

        # inductive split with graph task
        pyg_dataset = TUDataset("./enzymes", "ENZYMES")
        ds = pyg_to_dicts(pyg_dataset)
        graphs = [Graph(**item) for item in ds]
        num_graphs = len(graphs)
        split_ratio = [0.3, 0.3, 0.4]
        graph_size_list = []
        split_offset = 0
        custom_split_graphs = []
        for i, split_ratio_i in enumerate(split_ratio):
            if i != len(split_ratio) - 1:
                num_split_i = (
                    1 +
                    int(split_ratio_i * (num_graphs - len(split_ratio)))
                )
                custom_split_graphs.append(
                    graphs[split_offset: split_offset + num_split_i]
                )
                split_offset += num_split_i
                graph_size_list.append(num_split_i)
            else:
                custom_split_graphs.append(graphs[split_offset:])
                graph_size_list.append(len(graphs[split_offset:]))
        dataset = GraphDataset(
            graphs, task="graph",
            custom_split_graphs=custom_split_graphs
        )
        split_res = dataset.split(transductive=False)
        self.assertEqual(graph_size_list[0], len(split_res[0]))
        self.assertEqual(graph_size_list[1], len(split_res[1]))
        self.assertEqual(graph_size_list[2], len(split_res[2]))

    #     # transductive split with link_pred task in `disjoint` edge_train_mode.
    #     pyg_dataset = Planetoid("./cora", "Cora")
    #     graphs = GraphDataset.pyg_to_graphs(pyg_dataset)
    #     split_ratio = [0.3, 0.3, 0.4]
    #     link_size_list = [0 for i in range(len(split_ratio))]

    #     for graph in graphs:
    #         split_offset = 0
    #         edges = list(graph.G.edges)
    #         random.shuffle(edges)
    #         num_edges_train = 1 + int(split_ratio[0] * (graph.num_edges - 3))
    #         num_edges_train_disjoint = (
    #             1 + int(split_ratio[0] * 0.5 * (graph.num_edges - 3))
    #         )
    #         num_edges_val = 1 + int(split_ratio[0] * (graph.num_edges - 3))

    #         edges_train = edges[:num_edges_train]
    #         edges_train_disjoint = edges[:num_edges_train_disjoint]
    #         edges_val = edges[num_edges_train:num_edges_train + num_edges_val]
    #         edges_test = edges[num_edges_train + num_edges_val:]

    #         custom_splits = [
    #             edges_train,
    #             edges_val,
    #             edges_test,
    #         ]
    #         graph.custom = {
    #             "general_splits": custom_splits,
    #             "disjoint_split": edges_train_disjoint
    #         }

    #         link_size_list[0] += len(edges_train_disjoint)
    #         link_size_list[1] += len(edges_val)
    #         link_size_list[2] += len(edges_test)

    #     dataset = GraphDataset(
    #         graphs,
    #         task="link_pred",
    #         edge_train_mode="disjoint"
    #     )
    #     split_res = dataset.split(transductive=True)
    #     self.assertEqual(
    #         split_res[0][0].edge_label_index.shape[1],
    #         2 * 2 * link_size_list[0]
    #     )
    #     self.assertEqual(
    #         split_res[1][0].edge_label_index.shape[1],
    #         2 * 2 * link_size_list[1]
    #     )
    #     self.assertEqual(
    #         split_res[2][0].edge_label_index.shape[1],
    #         2 * 2 * link_size_list[2]
    #     )

    # def test_apply_transform(self):
    #     def transform_func(graph):
    #         G = graph.G
    #         for v in G.nodes:
    #             G.nodes[v]["node_feature"] = torch.ones(5)
    #         for u, v, edge_key in G.edges:
    #             edge_feature = G[u][v][edge_key]["edge_feature"]
    #             G[u][v][edge_key]["edge_feature"] = 2 * edge_feature
    #         graph.G = G
    #         return graph

    #     G, x, y, edge_x, edge_y, edge_index, graph_x, graph_y = (
    #         simple_networkx_multigraph()
    #     )
    #     Graph.add_edge_attr(G, "edge_feature", edge_x)
    #     Graph.add_edge_attr(G, "edge_label", edge_y)
    #     Graph.add_node_attr(G, "node_label", y)
    #     Graph.add_graph_attr(G, "graph_feature", graph_x)
    #     Graph.add_graph_attr(G, "graph_label", graph_y)

    #     graph = Graph(G)
    #     graphs = [graph]
    #     dataset = GraphDataset(
    #         graphs,
    #         task="link_pred",
    #         edge_train_mode="disjoint"
    #     )
    #     edge_feature = dataset[0].edge_feature

    #     dataset_transform = dataset.apply_transform(transform_func)

    #     self.assertEqual(
    #         torch.sum(
    #             dataset_transform[0].node_feature
    #             - torch.ones([G.number_of_nodes(), 5])
    #         ).item(),
    #         0
    #     )

    #     self.assertEqual(
    #         torch.sum(
    #             dataset_transform[0].edge_feature - 2 * edge_feature
    #         ).item(),
    #         0
    #     )

    # def test_generator(self):
    #     pyg_dataset = Planetoid("./cora", "Cora")
    #     dg = Graph.pyg_to_graph(pyg_dataset[0])

    #     num_nodes = 500
    #     sizes = [2, 3]

    #     class NeighborGenerator(Generator):
    #         def __len__(self):
    #             return sizes

    #         def generate(self):
    #             graph = Graph(gen_graph(num_nodes, dg.G))
    #             return graph

    #     dataset = GraphDataset(None, generator=NeighborGenerator(sizes))
    #     self.assertTrue(dataset[0].node_feature.shape[0] == num_nodes)

    # def test_ensemble_generator(self):
    #     pyg_dataset = Planetoid("./cora", "Cora")
    #     dg = Graph.pyg_to_graph(pyg_dataset[0])

    #     num_nodes = 500
    #     sizes = [2, 3]

    #     class NeighborGenerator1(Generator):
    #         def __len__(self):
    #             return sizes

    #         def generate(self):
    #             graph = Graph(gen_graph(num_nodes, dg.G))
    #             return graph

    #     class NeighborGenerator2(Generator):
    #         def __len__(self):
    #             return sizes

    #         def generate(self):
    #             graph = Graph(gen_graph(num_nodes, dg.G))
    #             return graph

    #     ensemble_generator = (
    #         EnsembleGenerator(
    #             [
    #                 NeighborGenerator1(sizes),
    #                 NeighborGenerator2(sizes),
    #             ]
    #         )
    #     )
    #     dataset = GraphDataset(None, generator=ensemble_generator)
    #     self.assertTrue(dataset[0].node_feature.shape[0] == num_nodes)

    def test_filter(self):
        pyg_dataset = TUDataset("./enzymes", "ENZYMES")
        ds = pyg_to_dicts(pyg_dataset)
        graphs = [Graph(**item) for item in ds]
        dataset = GraphDataset(graphs, task="graph")
        thresh = 90

        orig_dataset_size = len(dataset)
        num_graphs_large = 0
        for graph in dataset:
            if graph.num_nodes >= thresh:
                num_graphs_large += 1

        dataset = dataset.filter(
            lambda graph: graph.num_nodes < thresh, deep_copy=False
        )
        filtered_dataset_size = len(dataset)

        self.assertEqual(
            orig_dataset_size - filtered_dataset_size,
            num_graphs_large,
        )


if __name__ == "__main__":
    unittest.main()
