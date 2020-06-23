import torch
import unittest
from torch_geometric.datasets import TUDataset, Planetoid
from copy import deepcopy
from deepsnap.graph import Graph
from deepsnap.hetero_graph import HeteroGraph
from deepsnap.dataset import GraphDataset, Generator, EnsembleGenerator
from tests.utils import (
    simple_networkx_graph,
    generate_dense_hete_dataset,
    gen_graph,
)


class TestDataset(unittest.TestCase):

    def test_dataset_basic(self):
        G, x, y, edge_x, edge_y, edge_index, graph_x, graph_y = \
            simple_networkx_graph()
        Graph.add_edge_attr(G, "edge_feature", edge_x)
        Graph.add_edge_attr(G, "edge_label", edge_y)
        Graph.add_node_attr(G, "node_feature", x)
        Graph.add_node_attr(G, "node_label", y)
        Graph.add_graph_attr(G, "graph_feature", graph_x)
        Graph.add_graph_attr(G, "graph_label", graph_y)
        H = deepcopy(G)
        graphs = GraphDataset.list_to_graphs([G, H])
        dataset = GraphDataset(graphs)
        self.assertEqual(len(dataset), 2)

    def test_dataset_property(self):
        G, x, y, edge_x, edge_y, edge_index, graph_x, graph_y = \
            simple_networkx_graph()
        Graph.add_edge_attr(G, "edge_feature", edge_x)
        Graph.add_edge_attr(G, "edge_label", edge_y)
        Graph.add_node_attr(G, "node_feature", x)
        Graph.add_node_attr(G, "node_label", y)
        Graph.add_graph_attr(G, "graph_feature", graph_x)
        Graph.add_graph_attr(G, "graph_label", graph_y)
        H = G.copy()
        Graph.add_graph_attr(H, "graph_label", torch.tensor([1]))

        graphs = GraphDataset.list_to_graphs([G, H])
        dataset = GraphDataset(graphs)
        self.assertEqual(dataset.num_node_labels, 5)
        self.assertEqual(dataset.num_node_features, 2)
        self.assertEqual(dataset.num_edge_labels, 4)
        self.assertEqual(dataset.num_edge_features, 2)
        self.assertEqual(dataset.num_graph_labels, 2)
        self.assertEqual(dataset.num_graph_features, 2)
        self.assertEqual(dataset.num_labels, 5)  # node task
        dataset = GraphDataset(graphs, task='edge')
        self.assertEqual(dataset.num_labels, 4)
        dataset = GraphDataset(graphs, task='link_pred')
        self.assertEqual(dataset.num_labels, 4)
        dataset = GraphDataset(graphs, task='graph')
        self.assertEqual(dataset.num_labels, 2)

    def test_dataset_hetero_graph_split(self):
        G = generate_dense_hete_dataset()
        hete = HeteroGraph(G)
        # node
        dataset = GraphDataset([hete], task='node')
        split_res = dataset.split()
        for node_type in hete.node_label_index:
            num_nodes = int(len(hete.node_label_index[node_type]))
            num_nodes_reduced = num_nodes - 3
            node_0 = 1 + int(num_nodes_reduced * 0.8)
            node_1 = 1 + int(num_nodes_reduced * 0.1)
            node_2 = num_nodes - node_0 - node_1

            self.assertEqual(
                len(split_res[0][0].node_label_index[node_type]), node_0)

            self.assertEqual(
                len(split_res[1][0].node_label_index[node_type]), node_1)

            self.assertEqual(
                len(split_res[2][0].node_label_index[node_type]), node_2)

        # node with specified split type
        dataset = GraphDataset([hete], task='node')
        node_split_types = ['n1']
        split_res = dataset.split(split_types=node_split_types)
        for node_type in hete.node_label_index:
            if node_type in node_split_types:
                num_nodes = int(len(hete.node_label_index[node_type]))
                num_nodes_reduced = num_nodes - 3
                node_0 = 1 + int(num_nodes_reduced * 0.8)
                node_1 = 1 + int(num_nodes_reduced * 0.1)
                node_2 = num_nodes - node_0 - node_1
                self.assertEqual(
                    len(split_res[0][0].node_label_index[node_type]), node_0)

                self.assertEqual(
                    len(split_res[1][0].node_label_index[node_type]), node_1)

                self.assertEqual(
                    len(split_res[2][0].node_label_index[node_type]), node_2)
            else:
                num_nodes = int(len(hete.node_label_index[node_type]))
                self.assertEqual(
                    len(split_res[0][0].node_label_index[node_type]), num_nodes)

                self.assertEqual(
                    len(split_res[1][0].node_label_index[node_type]), num_nodes)

                self.assertEqual(
                    len(split_res[2][0].node_label_index[node_type]), num_nodes)

        # node with specified split type (string mode)
        dataset = GraphDataset([hete], task='node')
        node_split_types = 'n1'
        split_res = dataset.split(split_types=node_split_types)
        for node_type in hete.node_label_index:
            if node_type in node_split_types:
                num_nodes = int(len(hete.node_label_index[node_type]))
                num_nodes_reduced = num_nodes - 3
                node_0 = 1 + int(num_nodes_reduced * 0.8)
                node_1 = 1 + int(num_nodes_reduced * 0.1)
                node_2 = num_nodes - node_0 - node_1
                self.assertEqual(
                    len(split_res[0][0].node_label_index[node_type]), node_0)

                self.assertEqual(
                    len(split_res[1][0].node_label_index[node_type]), node_1)

                self.assertEqual(
                    len(split_res[2][0].node_label_index[node_type]), node_2)
            else:
                num_nodes = int(len(hete.node_label_index[node_type]))
                self.assertEqual(
                    len(split_res[0][0].node_label_index[node_type]), num_nodes)

                self.assertEqual(
                    len(split_res[1][0].node_label_index[node_type]), num_nodes)

                self.assertEqual(
                    len(split_res[2][0].node_label_index[node_type]), num_nodes)

        # edge
        dataset = GraphDataset([hete], task='edge')
        split_res = dataset.split()
        for edge_type in hete.edge_label_index:
            num_edges = hete.edge_label_index[edge_type].shape[1]
            num_edges_reduced = num_edges - 3
            edge_0 = 1 + int(num_edges_reduced * 0.8)
            edge_1 = 1 + int(num_edges_reduced * 0.1)
            edge_2 = num_edges - edge_0 - edge_1
            self.assertEqual(
                split_res[0][0].edge_label_index[edge_type].shape[1], edge_0)

            self.assertEqual(
                split_res[1][0].edge_label_index[edge_type].shape[1], edge_1)

            self.assertEqual(
                split_res[2][0].edge_label_index[edge_type].shape[1], edge_2)

        # edge with specified split type
        dataset = GraphDataset([hete], task='edge')
        edge_split_types = [('n1', 'e1', 'n1'), ('n1', 'e2', 'n2')]
        split_res = dataset.split(split_types=edge_split_types)
        for edge_type in hete.edge_label_index:
            if edge_type in edge_split_types:
                num_edges = hete.edge_label_index[edge_type].shape[1]
                num_edges_reduced = num_edges - 3
                edge_0 = 1 + int(num_edges_reduced * 0.8)
                edge_1 = 1 + int(num_edges_reduced * 0.1)
                edge_2 = num_edges - edge_0 - edge_1
                self.assertEqual(
                    split_res[0][0].edge_label_index[edge_type].shape[1], edge_0)

                self.assertEqual(
                    split_res[1][0].edge_label_index[edge_type].shape[1], edge_1)

                self.assertEqual(
                    split_res[2][0].edge_label_index[edge_type].shape[1], edge_2)
            else:
                num_edges = hete.edge_label_index[edge_type].shape[1]
                self.assertEqual(
                    split_res[0][0].edge_label_index[edge_type].shape[1], num_edges)

                self.assertEqual(
                    split_res[1][0].edge_label_index[edge_type].shape[1], num_edges)

                self.assertEqual(
                    split_res[2][0].edge_label_index[edge_type].shape[1], num_edges)

        # link_pred
        dataset = GraphDataset([hete], task='link_pred')
        split_res = dataset.split(transductive=True)
        for edge_type in hete.edge_label_index:
            num_edges = hete.edge_label_index[edge_type].shape[1]
            num_edges_reduced = num_edges - 3
            self.assertEqual(split_res[0][0].edge_label_index[edge_type].shape[1],
                             (2 * (1 + int(0.8 * (num_edges_reduced)))))
            self.assertEqual(split_res[1][0].edge_label_index[edge_type].shape[1],
                             (2 * (1 + (int(0.1 * (num_edges_reduced))))))
            self.assertEqual(split_res[2][0].edge_label_index[edge_type].shape[1],
                             2 * num_edges - 2 * (2 + int(0.1 * num_edges_reduced) +
                                                  int(0.8 * num_edges_reduced)))

        # link_pred with specified split type
        dataset = GraphDataset([hete], task='link_pred')
        link_split_types = [('n1', 'e1', 'n1'), ('n1', 'e2', 'n2')]
        split_res = dataset.split(transductive=True, split_types=link_split_types)

        for edge_type in hete.edge_label_index:
            if edge_type in link_split_types:
                num_edges = hete.edge_label_index[edge_type].shape[1]
                num_edges_reduced = num_edges - 3
                self.assertEqual(split_res[0][0].edge_label_index[edge_type].shape[1],
                                 (2 * (1 + int(0.8 * (num_edges_reduced)))))
                self.assertEqual(split_res[1][0].edge_label_index[edge_type].shape[1],
                                 (2 * (1 + (int(0.1 * (num_edges_reduced))))))
                self.assertEqual(split_res[2][0].edge_label_index[edge_type].shape[1],
                                 2 * num_edges - 2 * (2 + int(0.1 * num_edges_reduced) +
                                                      int(0.8 * num_edges_reduced)))
            else:
                num_edges = hete.edge_label_index[edge_type].shape[1]
                self.assertEqual(split_res[0][0].edge_label_index[edge_type].shape[1],
                                 (1 * (0 + int(1.0 * (num_edges)))))
                self.assertEqual(split_res[1][0].edge_label_index[edge_type].shape[1],
                                 (1 * (0 + (int(1.0 * (num_edges))))))
                self.assertEqual(split_res[2][0].edge_label_index[edge_type].shape[1],
                                 1 * (0 + (int(1.0 * (num_edges)))))

        # link_pred + disjoint
        dataset = GraphDataset([hete], task='link_pred', edge_train_mode='disjoint', edge_message_ratio=0.5)
        split_res = dataset.split(transductive=True, split_ratio=[0.6, 0.2, 0.2])
        for edge_type in hete.edge_label_index:
            num_edges = hete.edge_label_index[edge_type].shape[1]
            num_edges_reduced = num_edges - 3
            edge_0 = (1 + int(0.6 * num_edges_reduced))
            edge_0 = 2 * (edge_0 - (1 + int(0.5 * (edge_0 - 2))))

            self.assertEqual(split_res[0][0].edge_label_index[edge_type].shape[1], edge_0)
            edge_1 = 2 * (1 + int(0.2 * num_edges_reduced))
            self.assertEqual(split_res[1][0].edge_label_index[edge_type].shape[1], edge_1)
            edge_2 = 2 * int(num_edges) - \
                (2 * (1 + int(0.6 * num_edges_reduced))) - edge_1
            self.assertEqual(split_res[2][0].edge_label_index[edge_type].shape[1], edge_2)

        # link pred with edge_split_mode set to "exact"
        dataset = GraphDataset([hete], task='link_pred', edge_split_mode="approximate")
        split_res = dataset.split(transductive=True)
        hete_link_train_edge_num = 0
        hete_link_test_edge_num = 0
        hete_link_val_edge_num = 0
        num_edges = 0
        for edge_type in hete.edge_label_index:
            num_edges += hete.edge_label_index[edge_type].shape[1]
            if edge_type in split_res[0][0].edge_label_index:
                hete_link_train_edge_num += split_res[0][0].edge_label_index[edge_type].shape[1]
            if edge_type in split_res[1][0].edge_label_index:
                hete_link_test_edge_num += split_res[1][0].edge_label_index[edge_type].shape[1]
            if edge_type in split_res[2][0].edge_label_index:
                hete_link_val_edge_num += split_res[2][0].edge_label_index[edge_type].shape[1]

        num_edges_reduced = num_edges - 3
        self.assertEqual(hete_link_train_edge_num,
                         (2 * (1 + int(0.8 * (num_edges_reduced)))))
        self.assertEqual(hete_link_test_edge_num,
                         (2 * (1 + (int(0.1 * (num_edges_reduced))))))
        self.assertEqual(hete_link_val_edge_num,
                         2 * num_edges - 2 * (2 + int(0.1 * num_edges_reduced) +
                                              int(0.8 * num_edges_reduced)))

        # link pred with specified types and edge_split_mode set to "exact"
        dataset = GraphDataset([hete], task='link_pred', edge_split_mode="approximate")
        link_split_types = [('n1', 'e1', 'n1'), ('n1', 'e2', 'n2')]
        split_res = dataset.split(transductive=True, split_types=link_split_types)
        hete_link_train_edge_num = 0
        hete_link_test_edge_num = 0
        hete_link_val_edge_num = 0

        num_split_type_edges = 0
        num_non_split_type_edges = 0
        for edge_type in hete.edge_label_index:
            if edge_type in link_split_types:
                num_split_type_edges += hete.edge_label_index[edge_type].shape[1]
            else:
                num_non_split_type_edges += hete.edge_label_index[edge_type].shape[1]
            if edge_type in split_res[0][0].edge_label_index:
                hete_link_train_edge_num += split_res[0][0].edge_label_index[edge_type].shape[1]
            if edge_type in split_res[1][0].edge_label_index:
                hete_link_test_edge_num += split_res[1][0].edge_label_index[edge_type].shape[1]
            if edge_type in split_res[2][0].edge_label_index:
                hete_link_val_edge_num += split_res[2][0].edge_label_index[edge_type].shape[1]

        num_edges_reduced = num_split_type_edges - 3
        edge_0 = 2 * (1 + int(0.8 * (num_edges_reduced))) + num_non_split_type_edges
        edge_1 = 2 * (1 + int(0.1 * (num_edges_reduced))) + num_non_split_type_edges
        edge_2 = 2 * num_split_type_edges - 2 * (2 + int(0.1 * num_edges_reduced) + \
                                                 int(0.8 * num_edges_reduced)) + num_non_split_type_edges

        self.assertEqual(hete_link_train_edge_num, edge_0)
        self.assertEqual(hete_link_test_edge_num, edge_1)
        self.assertEqual(hete_link_val_edge_num, edge_2)

    def test_dataset_split(self):
        # inductively split with graph task
        pyg_dataset = TUDataset('./enzymes', 'ENZYMES')
        graphs = GraphDataset.pyg_to_graphs(pyg_dataset)
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
        pyg_dataset = TUDataset('./enzymes', 'ENZYMES')
        graphs = GraphDataset.pyg_to_graphs(pyg_dataset)
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
        pyg_dataset = TUDataset('./enzymes', 'ENZYMES')
        graphs = GraphDataset.pyg_to_graphs(pyg_dataset)
        dataset = GraphDataset(
            graphs, task="link_pred", edge_train_mode="disjoint")
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
        pyg_dataset = Planetoid('./cora', 'Cora')
        graphs = GraphDataset.pyg_to_graphs(pyg_dataset)
        dataset = GraphDataset(graphs, task="node")
        num_nodes = dataset.num_nodes[0]
        num_nodes_reduced = num_nodes - 3
        num_edges = dataset.num_edges[0]
        num_edges_reduced = num_edges - 3
        split_res = dataset.split()
        self.assertEqual(len(split_res[0][0].node_label_index),
                         1 + int(0.8 * num_nodes_reduced))
        self.assertEqual(len(split_res[1][0].node_label_index),
                         1 + int(0.1 * num_nodes_reduced))
        self.assertEqual(len(split_res[2][0].node_label_index), num_nodes - 2 -
                         int(0.8 * num_nodes_reduced) - int(0.1 * num_nodes_reduced))

        # transductively split with edge task
        dataset = GraphDataset(graphs, task="edge")
        split_res = dataset.split()
        edge_0 = 2 * (1 + int(0.8 * (num_edges_reduced)))
        self.assertEqual(split_res[0][0].edge_label_index.shape[1],
                         edge_0)
        edge_1 = 2 * (1 + int(0.1 * (num_edges_reduced)))
        self.assertEqual(split_res[1][0].edge_label_index.shape[1],
                         edge_1)
        self.assertEqual(split_res[2][0].edge_label_index.shape[1],
                         2 * num_edges - edge_0 - edge_1)

        # transductively split with link_pred task
        # and default (`all`) edge_train_mode
        dataset = GraphDataset(graphs, task="link_pred")
        split_res = dataset.split()
        self.assertEqual(split_res[0][0].edge_label_index.shape[1], 2 *
                         (2 * (1 + int(0.8 * (num_edges_reduced)))))
        self.assertEqual(split_res[1][0].edge_label_index.shape[1], 2 *
                         (2 * (1 + (int(0.1 * (num_edges_reduced))))))
        self.assertEqual(split_res[2][0].edge_label_index.shape[1],
                         2 * 2 * num_edges - 2 * 2 * (2 + int(0.1 * num_edges_reduced) +
                                                      int(0.8 * num_edges_reduced)))

        # transductively split with link_pred task, `split` edge_train_mode
        # and 0.5 edge_message_ratio
        dataset = GraphDataset(
            graphs, task="link_pred",
            edge_train_mode="disjoint",
            edge_message_ratio=0.5)
        split_res = dataset.split()
        edge_0 = 2 * (1 + int(0.8 * num_edges_reduced))
        edge_0 = 2 * (edge_0 - (1 + int(0.5 * (edge_0 - 3))))
        self.assertEqual(
            split_res[0][0].edge_label_index.shape[1],
            edge_0)
        edge_1 = 2 * 2 * (1 + int(0.1 * num_edges_reduced))
        self.assertEqual(split_res[1][0].edge_label_index.shape[1], edge_1)
        edge_2 = 2 * 2 * int(num_edges) - \
            2 * (2 * (1 + int(0.8 * num_edges_reduced))) - edge_1
        self.assertEqual(split_res[2][0].edge_label_index.shape[1], edge_2)

        # transductively split with link_pred task
        # and specified edge_negative_sampling_ratio
        dataset = GraphDataset(
            graphs, task="link_pred", edge_negative_sampling_ratio=2)
        split_res = dataset.split()
        edge_0 = (2 + 1) * (2 * (1 + int(0.8 * num_edges_reduced)))
        self.assertEqual(split_res[0][0].edge_label_index.shape[1], edge_0)
        edge_1 = (2 + 1) * 2 * (1 + int(0.1 * num_edges_reduced))
        self.assertEqual(split_res[1][0].edge_label_index.shape[1], edge_1)
        edge_2 = (2 + 1) * 2 * int(num_edges) - edge_0 - edge_1
        self.assertEqual(split_res[2][0].edge_label_index.shape[1], edge_2)

    def test_generator(self):
        pyg_dataset = Planetoid('./cora', 'Cora')
        dg = Graph.pyg_to_graph(pyg_dataset[0])

        num_nodes = 500
        sizes = [2, 3]

        class NeighborGenerator(Generator):
            def __len__(self):
                return sizes

            def generate(self):
                graph = Graph(gen_graph(num_nodes, dg.G))
                return graph

        dataset = GraphDataset(None, generator=NeighborGenerator(sizes))
        self.assertTrue(dataset[0].node_feature.shape[0] == num_nodes)

    def test_ensemble_generator(self):
        pyg_dataset = Planetoid('./cora', 'Cora')
        dg = Graph.pyg_to_graph(pyg_dataset[0])

        num_nodes = 500
        sizes = [2, 3]

        class NeighborGenerator1(Generator):
            def __len__(self):
                return sizes

            def generate(self):
                graph = Graph(gen_graph(num_nodes, dg.G))
                return graph

        class NeighborGenerator2(Generator):
            def __len__(self):
                return sizes

            def generate(self):
                graph = Graph(gen_graph(num_nodes, dg.G))
                return graph

        ensemble_generator = \
            EnsembleGenerator(
                [NeighborGenerator1(sizes), NeighborGenerator2(sizes)])
        dataset = GraphDataset(None, generator=ensemble_generator)
        self.assertTrue(dataset[0].node_feature.shape[0] == num_nodes)

    def test_filter(self):
        pyg_dataset = TUDataset('./enzymes', 'ENZYMES')
        graphs = GraphDataset.pyg_to_graphs(pyg_dataset)
        dataset = GraphDataset(graphs, task="graph")
        thresh = 90

        orig_dataset_size = len(dataset)
        num_graphs_large = 0
        for graph in dataset:
            if len(graph.G) >= thresh:
                num_graphs_large += 1

        dataset = dataset.filter(
            lambda graph: len(graph.G) < thresh, deep_copy=False)
        filtered_dataset_size = len(dataset)

        self.assertEqual(
            orig_dataset_size - filtered_dataset_size, num_graphs_large)


if __name__ == '__main__':
    unittest.main()
