import copy
import random
import torch
import unittest
from torch_geometric.datasets import TUDataset, Planetoid
from copy import deepcopy
from deepsnap.graph import Graph
from deepsnap.hetero_graph import HeteroGraph
from deepsnap.dataset import GraphDataset, Generator, EnsembleGenerator
from tests.utils import (
    pyg_to_dicts,
    simple_networkx_graph,
    simple_networkx_small_graph,
    simple_networkx_graph_alphabet,
    simple_networkx_multigraph,
    generate_dense_hete_dataset,
    generate_simple_small_hete_graph,
    gen_graph
)


class TestDatasetTensorBackend(unittest.TestCase):

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
        self.assertEqual(dataset.num_labels, 5)
        dataset = GraphDataset(graphs, task="graph")
        self.assertEqual(dataset.num_labels, 2)

    def test_dataset_hetero_graph_split(self):
        G = generate_dense_hete_dataset()
        hete = HeteroGraph(G)
        hete = HeteroGraph(
            node_feature=hete.node_feature,
            node_label=hete.node_label,
            edge_feature=hete.edge_feature,
            edge_label=hete.edge_label,
            edge_index=hete.edge_index,
            directed=True
        )

        # node
        dataset = GraphDataset([hete], task="node")
        split_res = dataset.split()
        for node_type in hete.node_label_index:
            num_nodes = int(len(hete.node_label_index[node_type]))
            node_0 = int(num_nodes * 0.8)
            node_1 = int(num_nodes * 0.1)
            node_2 = num_nodes - node_0 - node_1

            self.assertEqual(
                len(split_res[0][0].node_label_index[node_type]),
                node_0,
            )

            self.assertEqual(
                len(split_res[1][0].node_label_index[node_type]),
                node_1,
            )

            self.assertEqual(
                len(split_res[2][0].node_label_index[node_type]),
                node_2,
            )

        # node with specified split type
        dataset = GraphDataset([hete], task="node")
        node_split_types = ["n1"]
        split_res = dataset.split(split_types=node_split_types)
        for node_type in hete.node_label_index:
            if node_type in node_split_types:
                num_nodes = int(len(hete.node_label_index[node_type]))
                node_0 = int(num_nodes * 0.8)
                node_1 = int(num_nodes * 0.1)
                node_2 = num_nodes - node_0 - node_1
                self.assertEqual(
                    len(split_res[0][0].node_label_index[node_type]),
                    node_0,
                )

                self.assertEqual(
                    len(split_res[1][0].node_label_index[node_type]),
                    node_1,
                )

                self.assertEqual(
                    len(split_res[2][0].node_label_index[node_type]),
                    node_2,
                )
            else:
                num_nodes = int(len(hete.node_label_index[node_type]))
                self.assertEqual(
                    len(split_res[0][0].node_label_index[node_type]),
                    num_nodes,
                )

                self.assertEqual(
                    len(split_res[1][0].node_label_index[node_type]),
                    num_nodes,
                )

                self.assertEqual(
                    len(split_res[2][0].node_label_index[node_type]),
                    num_nodes,
                )

        # node with specified split type (string mode)
        dataset = GraphDataset([hete], task="node")
        node_split_types = "n1"
        split_res = dataset.split(split_types=node_split_types)
        for node_type in hete.node_label_index:
            if node_type in node_split_types:
                num_nodes = int(len(hete.node_label_index[node_type]))
                node_0 = int(num_nodes * 0.8)
                node_1 = int(num_nodes * 0.1)
                node_2 = num_nodes - node_0 - node_1
                self.assertEqual(
                    len(split_res[0][0].node_label_index[node_type]),
                    node_0,
                )

                self.assertEqual(
                    len(split_res[1][0].node_label_index[node_type]),
                    node_1,
                )

                self.assertEqual(
                    len(split_res[2][0].node_label_index[node_type]),
                    node_2,
                )
            else:
                num_nodes = int(len(hete.node_label_index[node_type]))
                self.assertEqual(
                    len(split_res[0][0].node_label_index[node_type]),
                    num_nodes,
                )

                self.assertEqual(
                    len(split_res[1][0].node_label_index[node_type]),
                    num_nodes,
                )

                self.assertEqual(
                    len(split_res[2][0].node_label_index[node_type]),
                    num_nodes,
                )

        # edge
        dataset = GraphDataset([hete], task="edge")
        split_res = dataset.split()
        for edge_type in hete.edge_label_index:
            num_edges = hete.edge_label_index[edge_type].shape[1]
            edge_0 = int(num_edges * 0.8)
            edge_1 = int(num_edges * 0.1)
            edge_2 = num_edges - edge_0 - edge_1
            self.assertEqual(
                split_res[0][0].edge_label_index[edge_type].shape[1],
                edge_0,
            )

            self.assertEqual(
                split_res[1][0].edge_label_index[edge_type].shape[1],
                edge_1,
            )

            self.assertEqual(
                split_res[2][0].edge_label_index[edge_type].shape[1],
                edge_2,
            )

        # edge with specified split type
        dataset = GraphDataset([hete], task="edge")
        edge_split_types = [("n1", "e1", "n1"), ("n1", "e2", "n2")]
        split_res = dataset.split(split_types=edge_split_types)
        for edge_type in hete.edge_label_index:
            if edge_type in edge_split_types:
                num_edges = hete.edge_label_index[edge_type].shape[1]
                edge_0 = int(num_edges * 0.8)
                edge_1 = int(num_edges * 0.1)
                edge_2 = num_edges - edge_0 - edge_1
                self.assertEqual(
                    split_res[0][0].edge_label_index[edge_type].shape[1],
                    edge_0,
                )

                self.assertEqual(
                    split_res[1][0].edge_label_index[edge_type].shape[1],
                    edge_1,
                )

                self.assertEqual(
                    split_res[2][0].edge_label_index[edge_type].shape[1],
                    edge_2,
                )
            else:
                num_edges = hete.edge_label_index[edge_type].shape[1]
                self.assertEqual(
                    split_res[0][0].edge_label_index[edge_type].shape[1],
                    num_edges,
                )

                self.assertEqual(
                    split_res[1][0].edge_label_index[edge_type].shape[1],
                    num_edges,
                )

                self.assertEqual(
                    split_res[2][0].edge_label_index[edge_type].shape[1],
                    num_edges,
                )

        # link_pred
        dataset = GraphDataset([hete], task="link_pred")
        split_res = dataset.split(transductive=True)
        for edge_type in hete.edge_label_index:
            num_edges = hete.edge_label_index[edge_type].shape[1]
            edge_0 = 2 * int(0.8 * num_edges)
            edge_1 = 2 * int(0.1 * num_edges)
            edge_2 = 2 * (
                num_edges - int(0.8 * num_edges) - int(0.1 * num_edges)
            )
            self.assertEqual(
                split_res[0][0].edge_label_index[edge_type].shape[1],
                edge_0
            )
            self.assertEqual(
                split_res[1][0].edge_label_index[edge_type].shape[1],
                edge_1
            )
            self.assertEqual(
                split_res[2][0].edge_label_index[edge_type].shape[1],
                edge_2
            )

        # link_pred with specified split type
        dataset = GraphDataset([hete], task="link_pred")
        link_split_types = [("n1", "e1", "n1"), ("n1", "e2", "n2")]
        split_res = dataset.split(
            transductive=True,
            split_types=link_split_types
        )

        for edge_type in hete.edge_label_index:
            if edge_type in link_split_types:
                num_edges = hete.edge_label_index[edge_type].shape[1]
                edge_0 = 2 * int(0.8 * num_edges)
                edge_1 = 2 * int(0.1 * num_edges)
                edge_2 = 2 * (
                    num_edges - int(0.8 * num_edges) - int(0.1 * num_edges)
                )
                self.assertEqual(
                    split_res[0][0].edge_label_index[edge_type].shape[1],
                    edge_0
                )
                self.assertEqual(
                    split_res[1][0].edge_label_index[edge_type].shape[1],
                    edge_1
                )
                self.assertEqual(
                    split_res[2][0].edge_label_index[edge_type].shape[1],
                    edge_2
                )
            else:
                num_edges = hete.edge_label_index[edge_type].shape[1]
                self.assertEqual(
                    split_res[0][0].edge_label_index[edge_type].shape[1],
                    num_edges
                )
                self.assertEqual(
                    split_res[1][0].edge_label_index[edge_type].shape[1],
                    num_edges
                )
                self.assertEqual(
                    split_res[2][0].edge_label_index[edge_type].shape[1],
                    num_edges
                )

        # link_pred + disjoint
        dataset = GraphDataset(
            [hete],
            task="link_pred",
            edge_train_mode="disjoint",
            edge_message_ratio=0.5,
        )
        split_res = dataset.split(
            transductive=True,
            split_ratio=[0.6, 0.2, 0.2],
        )
        for edge_type in hete.edge_label_index:
            num_edges = hete.edge_label_index[edge_type].shape[1]
            edge_0 = int(0.6 * num_edges)
            edge_0 = 2 * (edge_0 - int(0.5 * edge_0))
            edge_1 = 2 * int(0.2 * num_edges)
            edge_2 = 2 * (
                num_edges - int(0.6 * num_edges) - int(0.2 * num_edges)
            )

            self.assertEqual(
                split_res[0][0].edge_label_index[edge_type].shape[1],
                edge_0,
            )
            self.assertEqual(
                split_res[1][0].edge_label_index[edge_type].shape[1],
                edge_1,
            )
            self.assertEqual(
                split_res[2][0].edge_label_index[edge_type].shape[1],
                edge_2,
            )

        # link pred with edge_split_mode set to "exact"
        dataset = GraphDataset(
            [hete],
            task="link_pred",
            edge_split_mode="approximate"
        )
        split_res = dataset.split(transductive=True)
        hete_link_train_edge_num = 0
        hete_link_test_edge_num = 0
        hete_link_val_edge_num = 0
        num_edges = 0
        for edge_type in hete.edge_label_index:
            num_edges += hete.edge_label_index[edge_type].shape[1]
            if edge_type in split_res[0][0].edge_label_index:
                hete_link_train_edge_num += (
                    split_res[0][0].edge_label_index[edge_type].shape[1]
                )
            if edge_type in split_res[1][0].edge_label_index:
                hete_link_test_edge_num += (
                    split_res[1][0].edge_label_index[edge_type].shape[1]
                )
            if edge_type in split_res[2][0].edge_label_index:
                hete_link_val_edge_num += (
                    split_res[2][0].edge_label_index[edge_type].shape[1]
                )

        # num_edges_reduced = num_edges - 3
        edge_0 = 2 * int(0.8 * num_edges)
        edge_1 = 2 * int(0.1 * num_edges)
        edge_2 = 2 * (
            num_edges - int(0.8 * num_edges) - int(0.1 * num_edges)
        )

        self.assertEqual(
            hete_link_train_edge_num,
            edge_0
        )
        self.assertEqual(
            hete_link_test_edge_num,
            edge_1
        )
        self.assertEqual(
            hete_link_val_edge_num,
            edge_2
        )
        # link pred with specified types and edge_split_mode set to "exact"
        dataset = GraphDataset(
            [hete],
            task="link_pred",
            edge_split_mode="approximate",
        )
        link_split_types = [("n1", "e1", "n1"), ("n1", "e2", "n2")]
        split_res = dataset.split(
            transductive=True,
            split_types=link_split_types,
        )
        hete_link_train_edge_num = 0
        hete_link_test_edge_num = 0
        hete_link_val_edge_num = 0

        num_split_type_edges = 0
        num_non_split_type_edges = 0
        for edge_type in hete.edge_label_index:
            if edge_type in link_split_types:
                num_split_type_edges += (
                    hete.edge_label_index[edge_type].shape[1]
                )
            else:
                num_non_split_type_edges += (
                    hete.edge_label_index[edge_type].shape[1]
                )
            if edge_type in split_res[0][0].edge_label_index:
                hete_link_train_edge_num += (
                    split_res[0][0].edge_label_index[edge_type].shape[1]
                )
            if edge_type in split_res[1][0].edge_label_index:
                hete_link_test_edge_num += (
                    split_res[1][0].edge_label_index[edge_type].shape[1]
                )
            if edge_type in split_res[2][0].edge_label_index:
                hete_link_val_edge_num += (
                    split_res[2][0].edge_label_index[edge_type].shape[1]
                )

        # num_edges_reduced = num_split_type_edges - 3
        num_edges = num_split_type_edges
        edge_0 = 2 * int(0.8 * num_edges) + num_non_split_type_edges
        edge_1 = 2 * int(0.1 * num_edges) + num_non_split_type_edges
        edge_2 = 2 * (
            num_edges - int(0.8 * num_edges) - int(0.1 * num_edges)
        ) + num_non_split_type_edges

        self.assertEqual(hete_link_train_edge_num, edge_0)
        self.assertEqual(hete_link_test_edge_num, edge_1)
        self.assertEqual(hete_link_val_edge_num, edge_2)

    def test_dataset_split(self):
        # inductively split with graph task
        pyg_dataset = TUDataset("./enzymes", "ENZYMES")
        ds = pyg_to_dicts(pyg_dataset)
        graphs = [Graph(**item) for item in ds]
        dataset = GraphDataset(graphs, task="graph")
        split_res = dataset.split(transductive=False)
        num_graphs = len(dataset)
        num_train = int(0.8 * num_graphs)
        num_val = int(0.1 * num_graphs)
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
        num_train = int(0.8 * num_graphs)
        num_val = int(0.1 * num_graphs)
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
        num_train = int(0.8 * num_graphs)
        num_val = int(0.1 * num_graphs)
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
        num_edges = dataset.num_edges[0]
        node_0 = int(0.8 * num_nodes)
        node_1 = int(0.1 * num_nodes)
        node_2 = num_nodes - node_0 - node_1
        split_res = dataset.split()
        self.assertEqual(
            len(split_res[0][0].node_label_index),
            node_0
        )
        self.assertEqual(
            len(split_res[1][0].node_label_index),
            node_1
        )
        self.assertEqual(
            len(split_res[2][0].node_label_index),
            node_2
        )

        # transductively split with link_pred task
        # and default (`all`) edge_train_mode
        dataset = GraphDataset(graphs, task="link_pred")
        edge_0 = 2 * 2 * int(0.8 * num_edges)
        edge_1 = 2 * 2 * int(0.1 * num_edges)
        edge_2 = 2 * 2 * (
            num_edges - int(0.8 * num_edges) - int(0.1 * num_edges)
        )
        split_res = dataset.split()
        self.assertEqual(
            split_res[0][0].edge_label_index.shape[1],
            edge_0
        )
        self.assertEqual(
            split_res[1][0].edge_label_index.shape[1],
            edge_1
        )
        self.assertEqual(
            split_res[2][0].edge_label_index.shape[1],
            edge_2
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
        edge_0 = 2 * int(0.8 * num_edges)
        edge_0 = 2 * (edge_0 - int(0.5 * edge_0))
        edge_1 = 2 * 2 * int(0.1 * num_edges)
        edge_2 = 2 * 2 * (
            num_edges - int(0.8 * num_edges) - int(0.1 * num_edges)
        )
        self.assertEqual(
            split_res[0][0].edge_label_index.shape[1],
            edge_0,
        )
        self.assertEqual(split_res[1][0].edge_label_index.shape[1], edge_1)
        self.assertEqual(split_res[2][0].edge_label_index.shape[1], edge_2)

        # transductively split with link_pred task
        # and specified edge_negative_sampling_ratio
        dataset = GraphDataset(
            graphs,
            task="link_pred",
            edge_negative_sampling_ratio=2
        )
        split_res = dataset.split()
        edge_0 = (2 + 1) * (2 * int(0.8 * num_edges))
        edge_1 = (2 + 1) * (2 * int(0.1 * num_edges))
        edge_2 = (2 + 1) * (
            2 * (num_edges - int(0.8 * num_edges) - int(0.1 * num_edges))
        )

        self.assertEqual(split_res[0][0].edge_label_index.shape[1], edge_0)
        self.assertEqual(split_res[1][0].edge_label_index.shape[1], edge_1)
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
        nodes_train = torch.tensor(list(G.nodes)[: int(0.3 * num_nodes)])
        nodes_val = torch.tensor(
            list(G.nodes)[int(0.3 * num_nodes): int(0.6 * num_nodes)]
        )
        nodes_test = torch.tensor(list(G.nodes)[int(0.6 * num_nodes):])

        graph_train = Graph(
            node_feature=x, node_label=y, edge_index=edge_index,
            node_label_index=nodes_train, directed=True
        )
        graph_val = Graph(
            node_feature=x, node_label=y, edge_index=edge_index,
            node_label_index=nodes_val, directed=True
        )
        graph_test = Graph(
            node_feature=x, node_label=y, edge_index=edge_index,
            node_label_index=nodes_test, directed=True
        )

        graphs_train = [graph_train]
        graphs_val = [graph_val]
        graphs_test = [graph_test]

        dataset_train, dataset_val, dataset_test = (
            GraphDataset(graphs_train, task='node'),
            GraphDataset(graphs_val, task='node'),
            GraphDataset(graphs_test, task='node')
        )

        self.assertEqual(
            dataset_train[0].node_label_index.tolist(),
            list(range(int(0.3 * num_nodes)))
        )
        self.assertEqual(
            dataset_val[0].node_label_index.tolist(),
            list(range(int(0.3 * num_nodes), int(0.6 * num_nodes)))
        )
        self.assertEqual(
            dataset_test[0].node_label_index.tolist(),
            list(range(int(0.6 * num_nodes), num_nodes))
        )

        # transductive split with link_pred task (train/val split)
        edges = list(G.edges)
        num_edges = len(edges)
        edges_train = edges[: int(0.7 * num_edges)]
        edges_val = edges[int(0.7 * num_edges):]
        link_size_list = [len(edges_train), len(edges_val)]

        # generate pseudo pos and neg edges, they may overlap here
        train_pos = torch.LongTensor(edges_train).permute(1, 0)
        val_pos = torch.LongTensor(edges_val).permute(1, 0)
        val_neg = torch.randint(high=10, size=val_pos.shape, dtype=torch.int64)
        val_neg_double = torch.cat((val_neg, val_neg), dim=1)

        num_train = len(edges_train)
        num_val = len(edges_val)

        graph_train = Graph(
            node_feature=x, edge_index=edge_index,
            edge_feature=edge_x, directed=True,
            edge_label_index=train_pos
        )

        graph_val = Graph(
            node_feature=x, edge_index=edge_index,
            edge_feature=edge_x, directed=True,
            edge_label_index=val_pos,
            negative_edge=val_neg_double
        )

        graphs_train = [graph_train]
        graphs_val = [graph_val]

        dataset_train, dataset_val = (
            GraphDataset(
                graphs_train, task='link_pred', resample_negatives=True
            ),
            GraphDataset(
                graphs_val, task='link_pred', edge_negative_sampling_ratio=2
            )
        )

        self.assertEqual(
            dataset_train[0].edge_label_index.shape[1],
            2 * link_size_list[0]
        )
        self.assertEqual(
            dataset_train[0].edge_label.shape[0],
            2 * link_size_list[0]
        )
        self.assertEqual(
            dataset_val[0].edge_label_index.shape[1],
            val_pos.shape[1] + val_neg_double.shape[1]
        )
        self.assertEqual(
            dataset_val[0].edge_label.shape[0],
            val_pos.shape[1] + val_neg_double.shape[1]
        )
        self.assertTrue(
            torch.equal(
                dataset_train[0].edge_label_index[:, :num_train],
                train_pos
            )
        )
        self.assertTrue(
            torch.equal(
                dataset_val[0].edge_label_index[:, :num_val],
                val_pos
            )
        )
        self.assertTrue(
            torch.equal(
                dataset_val[0].edge_label_index[:, num_val:],
                val_neg_double
            )
        )

        dataset_train.resample_negatives = False
        self.assertTrue(
            torch.equal(
                dataset_train[0].edge_label_index,
                dataset_train[0].edge_label_index
            )
        )

        # transductive split with link_pred task with edge label
        edge_label_train = torch.LongTensor([1, 2, 3, 2, 1, 1, 2, 3, 2, 0, 0])
        edge_label_val = torch.LongTensor([1, 2, 3, 2, 1, 0])

        graph_train = Graph(
            node_feature=x,
            edge_index=edge_index,
            directed=True,
            edge_label_index=train_pos,
            edge_label=edge_label_train
        )

        graph_val = Graph(
            node_feature=x,
            edge_index=edge_index,
            directed=True,
            edge_label_index=val_pos,
            negative_edge=val_neg,
            edge_label=edge_label_val
        )

        graphs_train = [graph_train]
        graphs_val = [graph_val]

        dataset_train, dataset_val = (
            GraphDataset(graphs_train, task='link_pred'),
            GraphDataset(graphs_val, task='link_pred')
        )

        self.assertTrue(
            torch.equal(
                dataset_train[0].edge_label_index,
                dataset_train[0].edge_label_index
            )
        )

        self.assertTrue(
            torch.equal(
                dataset_train[0].edge_label[:num_train],
                edge_label_train
            )
        )

        self.assertTrue(
            torch.equal(
                dataset_val[0].edge_label[:num_val],
                edge_label_val
            )
        )

        # Multiple graph tensor backend link prediction (inductive)
        pyg_dataset = Planetoid('./cora', 'Cora')
        x = pyg_dataset[0].x
        y = pyg_dataset[0].y
        edge_index = pyg_dataset[0].edge_index
        row, col = edge_index
        mask = row < col
        row, col = row[mask], col[mask]
        edge_index = torch.stack([row, col], dim=0)
        edge_index = torch.cat(
            [edge_index, torch.flip(edge_index, [0])], dim=1
        )

        graphs = [
            Graph(
                node_feature=x, node_label=y,
                edge_index=edge_index, directed=False
            )
        ]
        graphs = [copy.deepcopy(graphs[0]) for _ in range(10)]

        edge_label_index = graphs[0].edge_label_index
        dataset = GraphDataset(
            graphs,
            task='link_pred',
            edge_message_ratio=0.6,
            edge_train_mode="all"
        )
        datasets = {}
        datasets['train'], datasets['val'], datasets['test'] = dataset.split(
            transductive=False, split_ratio=[0.85, 0.05, 0.1]
        )
        edge_label_index_split = (
            datasets['train'][0].edge_label_index[
                :, 0:edge_label_index.shape[1]
            ]
        )

        self.assertTrue(
            torch.equal(
                edge_label_index,
                edge_label_index_split
            )
        )

        # transductive split with node task (pytorch geometric dataset)
        pyg_dataset = Planetoid("./cora", "Cora")
        ds = pyg_to_dicts(pyg_dataset, task="cora")
        graphs = [Graph(**item) for item in ds]
        split_ratio = [0.3, 0.3, 0.4]
        node_size_list = [0 for i in range(len(split_ratio))]
        for graph in graphs:
            custom_splits = [[] for i in range(len(split_ratio))]
            split_offset = 0
            num_nodes = graph.num_nodes
            shuffled_node_indices = torch.randperm(graph.num_nodes)
            for i, split_ratio_i in enumerate(split_ratio):
                if i != len(split_ratio) - 1:
                    num_split_i = int(split_ratio_i * num_nodes)
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

        node_feature = graphs[0].node_feature
        edge_index = graphs[0].edge_index
        directed = graphs[0].directed

        graph_train = Graph(
            node_feature=node_feature,
            edge_index=edge_index,
            directed=directed,
            node_label_index=graphs[0].custom["general_splits"][0]
        )

        graph_val = Graph(
            node_feature=node_feature,
            edge_index=edge_index,
            directed=directed,
            node_label_index=graphs[0].custom["general_splits"][1]
        )

        graph_test = Graph(
            node_feature=node_feature,
            edge_index=edge_index,
            directed=directed,
            node_label_index=graphs[0].custom["general_splits"][2]
        )

        train_dataset = GraphDataset([graph_train], task="node")
        val_dataset = GraphDataset([graph_val], task="node")
        test_dataset = GraphDataset([graph_test], task="node")

        self.assertEqual(
            len(train_dataset[0].node_label_index),
            node_size_list[0]
        )
        self.assertEqual(
            len(val_dataset[0].node_label_index),
            node_size_list[1]
        )
        self.assertEqual(
            len(test_dataset[0].node_label_index),
            node_size_list[2]
        )

        # transductive split with edge task
        pyg_dataset = Planetoid("./cora", "Cora")
        graphs_g = GraphDataset.pyg_to_graphs(pyg_dataset)
        ds = pyg_to_dicts(pyg_dataset, task="cora")
        graphs = [Graph(**item) for item in ds]
        split_ratio = [0.3, 0.3, 0.4]
        edge_size_list = [0 for i in range(len(split_ratio))]
        for i, graph in enumerate(graphs):
            custom_splits = [[] for i in range(len(split_ratio))]
            split_offset = 0
            edges = list(graphs_g[i].G.edges)
            num_edges = graph.num_edges
            random.shuffle(edges)
            for i, split_ratio_i in enumerate(split_ratio):
                if i != len(split_ratio) - 1:
                    num_split_i = int(split_ratio_i * num_edges)
                    edges_split_i = (
                        edges[split_offset: split_offset + num_split_i]
                    )
                    split_offset += num_split_i
                else:
                    edges_split_i = edges[split_offset:]

                custom_splits[i] = edges_split_i
                edge_size_list[i] += len(edges_split_i)
            graph.custom = {
                "general_splits": custom_splits
            }

        node_feature = graphs[0].node_feature
        edge_index = graphs[0].edge_index
        directed = graphs[0].directed

        train_index = torch.tensor(
            graphs[0].custom["general_splits"][0]
        ).permute(1, 0)
        train_index = torch.cat((train_index, train_index), dim=1)
        val_index = torch.tensor(
            graphs[0].custom["general_splits"][1]
        ).permute(1, 0)
        val_index = torch.cat((val_index, val_index), dim=1)
        test_index = torch.tensor(
            graphs[0].custom["general_splits"][2]
        ).permute(1, 0)
        test_index = torch.cat((test_index, test_index), dim=1)

        graph_train = Graph(
            node_feature=node_feature,
            edge_index=edge_index,
            directed=directed,
            edge_label_index=train_index
        )

        graph_val = Graph(
            node_feature=node_feature,
            edge_index=edge_index,
            directed=directed,
            edge_label_index=val_index
        )

        graph_test = Graph(
            node_feature=node_feature,
            edge_index=edge_index,
            directed=directed,
            edge_label_index=test_index
        )

        train_dataset = GraphDataset([graph_train], task="edge")
        val_dataset = GraphDataset([graph_val], task="edge")
        test_dataset = GraphDataset([graph_test], task="edge")

        self.assertEqual(
            train_dataset[0].edge_label_index.shape[1],
            2 * edge_size_list[0]
        )
        self.assertEqual(
            val_dataset[0].edge_label_index.shape[1],
            2 * edge_size_list[1]
        )
        self.assertEqual(
            test_dataset[0].edge_label_index.shape[1],
            2 * edge_size_list[2]
        )

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
                num_split_i = int(split_ratio_i * num_graphs)
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

    def test_resample_disjoint_heterogeneous(self):
        G = generate_dense_hete_dataset()
        hete = HeteroGraph(G)
        hete = HeteroGraph(
            node_feature=hete.node_feature,
            node_label=hete.node_label,
            edge_feature=hete.edge_feature,
            edge_label=hete.edge_label,
            edge_index=hete.edge_index,
            directed=True
        )
        graphs = [hete]
        dataset = GraphDataset(
            graphs,
            task="link_pred",
            edge_train_mode="disjoint",
            edge_message_ratio=0.8,
            resample_disjoint=True,
            resample_disjoint_period=1
        )
        dataset_train, _, _ = dataset.split(split_ratio=[0.5, 0.2, 0.3])
        graph_train_first = dataset_train[0]
        graph_train_second = dataset_train[0]

        for message_type in graph_train_first.edge_index:
            self.assertEqual(
                graph_train_first.edge_label_index[message_type].shape[1],
                graph_train_second.edge_label_index[message_type].shape[1]
            )
            self.assertEqual(
                graph_train_first.edge_label[message_type].shape,
                graph_train_second.edge_label[message_type].shape
            )

    def test_resample_disjoint(self):
        pyg_dataset = Planetoid("./cora", "Cora")
        graphs = GraphDataset.pyg_to_graphs(pyg_dataset)
        graph = graphs[0]
        graph = Graph(
            node_label=graph.node_label,
            node_feature=graph.node_feature,
            edge_index=graph.edge_index,
            edge_feature=graph.edge_feature,
            directed=False
        )
        graphs = [graph]
        dataset = GraphDataset(
            graphs,
            task="link_pred",
            edge_train_mode="disjoint",
            edge_message_ratio=0.8,
            resample_disjoint=True,
            resample_disjoint_period=1
        )
        dataset_train, _, _ = dataset.split(split_ratio=[0.5, 0.2, 0.3])
        graph_train_first = dataset_train[0]
        graph_train_second = dataset_train[0]

        self.assertEqual(
            graph_train_first.edge_label_index.shape[1],
            graph_train_second.edge_label_index.shape[1]
        )
        self.assertTrue(
            torch.equal(
                graph_train_first.edge_label,
                graph_train_second.edge_label
            )
        )

    def test_secure_split_heterogeneous(self):
        G = generate_simple_small_hete_graph()
        graph = HeteroGraph(G)
        graph = HeteroGraph(
            node_label=graph.node_label,
            edge_index=graph.edge_index,
            edge_label=graph.edge_label,
            directed=True
        )
        graphs = [graph]

        # node task
        dataset = GraphDataset(graphs, task="node")
        split_res = dataset.split()
        for node_type in graph.node_label_index:
            num_nodes = graph.node_label_index[node_type].shape[0]
            num_nodes_reduced = num_nodes - 3
            node_0 = 1 + int(num_nodes_reduced * 0.8)
            node_1 = 1 + int(num_nodes_reduced * 0.1)
            node_2 = num_nodes - node_0 - node_1
            node_size = [node_0, node_1, node_2]
            for i in range(3):
                self.assertEqual(
                    split_res[i][0].node_label_index[node_type].shape[0],
                    node_size[i]
                )
                self.assertEqual(
                    split_res[i][0].node_label[node_type].shape[0],
                    node_size[i]
                )

        # edge task
        dataset = GraphDataset(graphs, task="edge")
        split_res = dataset.split()
        for message_type in graph.edge_label_index:
            num_edges = graph.edge_label_index[message_type].shape[1]
            num_edges_reduced = num_edges - 3
            edge_0 = 1 + int(num_edges_reduced * 0.8)
            edge_1 = 1 + int(num_edges_reduced * 0.1)
            edge_2 = num_edges - edge_0 - edge_1
            edge_size = [edge_0, edge_1, edge_2]
            for i in range(3):
                self.assertEqual(
                    split_res[i][0].edge_label_index[message_type].shape[1],
                    edge_size[i]
                )
                self.assertEqual(
                    split_res[i][0].edge_label[message_type].shape[0],
                    edge_size[i]
                )

        # link_pred task
        dataset = GraphDataset(graphs, task="link_pred")
        split_res = dataset.split()
        for message_type in graph.edge_label_index:
            num_edges = graph.edge_label_index[message_type].shape[1]
            num_edges_reduced = num_edges - 3
            edge_0 = 2 * (1 + int(num_edges_reduced * 0.8))
            edge_1 = 2 * (1 + int(num_edges_reduced * 0.1))
            edge_2 = 2 * num_edges - edge_0 - edge_1
            edge_size = [edge_0, edge_1, edge_2]
            for i in range(3):
                self.assertEqual(
                    split_res[i][0].edge_label_index[message_type].shape[1],
                    edge_size[i]
                )
                self.assertEqual(
                    split_res[i][0].edge_label[message_type].shape[0],
                    edge_size[i]
                )

    def test_secure_split(self):
        G = simple_networkx_small_graph()
        graph = Graph(G)
        graph = Graph(
            node_label=graph.node_label,
            edge_index=graph.edge_index,
            edge_label=graph.edge_label,
            directed=True
        )
        graphs = [graph]

        # node task
        dataset = GraphDataset(graphs, task="node")
        num_nodes = dataset.num_nodes[0]
        num_nodes_reduced = num_nodes - 3
        node_0 = 1 + int(0.8 * num_nodes_reduced)
        node_1 = 1 + int(0.1 * num_nodes_reduced)
        node_2 = num_nodes - node_0 - node_1
        node_size = [node_0, node_1, node_2]

        split_res = dataset.split()
        for i in range(3):
            self.assertEqual(
                split_res[i][0].node_label_index.shape[0],
                node_size[i]
            )
            self.assertEqual(
                split_res[i][0].node_label.shape[0],
                node_size[i]
            )

        # edge task
        dataset = GraphDataset(graphs, task="edge")
        num_edges = dataset.num_edges[0]
        num_edges_reduced = num_edges - 3
        edge_0 = 1 + int(0.8 * num_edges_reduced)
        edge_1 = 1 + int(0.1 * num_edges_reduced)
        edge_2 = num_edges - edge_0 - edge_1
        edge_size = [edge_0, edge_1, edge_2]

        split_res = dataset.split()
        for i in range(3):
            self.assertEqual(
                split_res[i][0].edge_label_index.shape[1],
                edge_size[i]
            )
            self.assertEqual(
                split_res[i][0].edge_label.shape[0],
                edge_size[i]
            )

        # link_pred task
        dataset = GraphDataset(graphs, task="link_pred")
        num_edges = dataset.num_edges[0]
        num_edges_reduced = num_edges - 3
        edge_0 = 2 * (1 + int(0.8 * num_edges_reduced))
        edge_1 = 2 * (1 + int(0.1 * num_edges_reduced))
        edge_2 = 2 * num_edges - edge_0 - edge_1
        edge_size = [edge_0, edge_1, edge_2]

        split_res = dataset.split()
        for i in range(3):
            self.assertEqual(
                split_res[i][0].edge_label_index.shape[1],
                edge_size[i]
            )
            self.assertEqual(
                split_res[i][0].edge_label.shape[0],
                edge_size[i]
            )

        # graph task
        graphs = [deepcopy(graph) for _ in range(5)]
        dataset = GraphDataset(graphs, task="link_pred")
        num_graphs = len(dataset)
        num_graphs_reduced = num_graphs - 3
        num_train = 1 + int(num_graphs_reduced * 0.8)
        num_val = 1 + int(num_graphs_reduced * 0.1)
        num_test = num_graphs - num_train - num_val
        split_res = dataset.split(transductive=False)
        self.assertEqual(num_train, len(split_res[0]))
        self.assertEqual(num_val, len(split_res[1]))
        self.assertEqual(num_test, len(split_res[2]))


if __name__ == "__main__":
    unittest.main()
