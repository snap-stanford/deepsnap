import torch
import unittest
import numpy as np
from tests.utils import simple_networkx_graph
from deepsnap.graph import Graph
from torch_geometric.datasets import Planetoid


class TestGraph(unittest.TestCase):

    def test_add_feature_nx(self):
        G, x, y, edge_x, edge_y, edge_index, graph_x, graph_y = (
            simple_networkx_graph()
        )
        Graph.add_edge_attr(G, "edge_feature", edge_x)
        Graph.add_edge_attr(G, "edge_label", edge_y)
        Graph.add_node_attr(G, "node_feature", x)
        Graph.add_node_attr(G, "node_label", y)
        Graph.add_graph_attr(G, "graph_feature", graph_x)
        Graph.add_graph_attr(G, "graph_label", graph_y)

        self.assertEqual(len(G.edges.data()), edge_index.shape[1])
        for item in G.edges.data():
            self.assertEqual('edge_feature' in item[2], True)
            self.assertEqual('edge_label' in item[2], True)
            self.assertEqual(len(item[2]['edge_feature']), 2)
            self.assertEqual(type(item[2]['edge_label'].item()), int)

        for item in G.nodes.data():
            self.assertEqual('node_feature' in item[1], True)
            self.assertEqual('node_label' in item[1], True)
            self.assertEqual(len(item[1]['node_feature']), 2)
            self.assertEqual(type(item[1]['node_label'].item()), int)

        self.assertEqual(
            G.graph.get("graph_feature").eq(graph_x).sum().item(),
            2,
        )
        self.assertEqual(
            G.graph.get("graph_label").eq(graph_y).sum().item(),
            1,
        )

    def test_graph_basics(self):
        G, x, y, edge_x, edge_y, edge_index, graph_x, graph_y = (
            simple_networkx_graph()
        )
        Graph.add_edge_attr(G, "edge_feature", edge_x)
        Graph.add_edge_attr(G, "edge_label", edge_y)
        Graph.add_node_attr(G, "node_feature", x)
        Graph.add_node_attr(G, "node_label", y)
        Graph.add_graph_attr(G, "graph_feature", graph_x)
        Graph.add_graph_attr(G, "graph_label", graph_y)
        dg = Graph(G)
        self.assertTrue(dg.is_directed())
        self.assertEqual(dg.is_undirected(), False)
        self.assertEqual(len(dg), 10)
        for item in [
                'G',
                'node_feature',
                'node_label',
                'edge_feature',
                'edge_label',
                'graph_feature',
                'graph_label',
                'edge_index',
                'edge_label_index',
                'node_label_index',
        ]:
            self.assertEqual(item in dg, True)
        self.assertEqual(len([key for key in dg]), 10)

        dg_t = Graph(node_feature=x, node_label=y, edge_index=edge_index, 
                     edge_feature=edge_x, edge_label=edge_y,
                     graph_feature=graph_x, graph_label=graph_y, directed=True)

        self.assertTrue(torch.equal(dg_t.node_feature, dg.node_feature))
        self.assertTrue(torch.equal(dg_t.node_label, dg.node_label))
        self.assertTrue(torch.equal(dg_t.edge_index, dg.edge_index))
        self.assertTrue(torch.equal(dg_t.edge_feature, dg.edge_feature))
        self.assertTrue(torch.equal(dg_t.edge_label, dg.edge_label))
        self.assertTrue(torch.equal(dg_t.graph_feature, dg.graph_feature))
        self.assertTrue(torch.equal(dg_t.graph_label, dg.graph_label))

        for item in [
                'directed',
                'node_feature',
                'node_label',
                'edge_feature',
                'edge_label',
                'graph_feature',
                'graph_label',
                'edge_index',
                'edge_label_index',
                'node_label_index',
        ]:
            self.assertEqual(item in dg_t, True)
        self.assertEqual(len([key for key in dg_t]), 10)

    def test_graph_property_edge_case(self):
        G_1, x, y, edge_x, edge_y, edge_index, graph_x, graph_y = (
            simple_networkx_graph()
        )
        dg_1 = Graph(G_1)
        self.assertEqual(dg_1.num_nodes, G_1.number_of_nodes())
        self.assertEqual(dg_1.num_edges, G_1.number_of_edges())
        self.assertEqual(dg_1.num_node_features, 0)
        self.assertEqual(dg_1.num_edge_features, 0)
        self.assertEqual(dg_1.num_graph_features, 0)
        self.assertEqual(dg_1.num_node_labels, 0)
        self.assertEqual(dg_1.num_edge_labels, 0)
        self.assertEqual(dg_1.num_graph_labels, 0)

        # Current tensor backend does not support num_nodes
        # without specifying any node related property

        G_2, x, y, edge_x, edge_y, edge_index, graph_x, graph_y = (
            simple_networkx_graph()
        )
        Graph.add_edge_attr(
            G_2,
            "edge_label",
            edge_y.type(torch.FloatTensor)
        )
        Graph.add_node_attr(
            G_2,
            "node_label",
            y.type(torch.FloatTensor)
        )
        Graph.add_graph_attr(
            G_2,
            "graph_label",
            graph_y.type(torch.FloatTensor)
        )

        dg_2 = Graph(G_2)
        self.assertEqual(dg_2.num_node_labels, 1)
        self.assertEqual(dg_2.num_edge_labels, 1)
        self.assertEqual(dg_2.num_graph_labels, 1)

        dg_2_t = Graph(node_label=y.type(torch.FloatTensor), edge_index=edge_index, 
                       edge_label=edge_y.type(torch.FloatTensor), 
                       graph_label=graph_y.type(torch.FloatTensor), directed=True)
        self.assertEqual(dg_2_t.num_node_labels, 1)
        self.assertEqual(dg_2_t.num_edge_labels, 1)
        self.assertEqual(dg_2_t.num_graph_labels, 1)

    def test_graph_property_general(self):
        G, x, y, edge_x, edge_y, edge_index, graph_x, graph_y = (
            simple_networkx_graph()
        )
        Graph.add_edge_attr(G, "edge_feature", edge_x)
        Graph.add_edge_attr(G, "edge_label", edge_y)
        Graph.add_node_attr(G, "node_feature", x)
        Graph.add_node_attr(G, "node_label", y)
        Graph.add_graph_attr(G, "graph_feature", graph_x)
        Graph.add_graph_attr(G, "graph_label", graph_y)

        dg = Graph(G)
        self.assertEqual(
            dg.keys,
            [
                'G',
                'node_feature',
                'node_label',
                'edge_feature',
                'edge_label',
                'graph_feature',
                'graph_label',
                'edge_index',
                'edge_label_index',
                'node_label_index',
            ]
        )
        self.assertEqual(dg.num_nodes, G.number_of_nodes())
        self.assertEqual(dg.num_edges, G.number_of_edges())
        self.assertEqual(dg.num_node_features, 2)
        self.assertEqual(dg.num_edge_features, 2)
        self.assertEqual(dg.num_graph_features, 2)
        self.assertEqual(dg.num_node_labels, np.max(y.data.numpy()) + 1)
        self.assertEqual(dg.num_edge_labels, np.max(edge_y.data.numpy()) + 1)
        self.assertEqual(dg.num_graph_labels, np.max(graph_y.data.numpy()) + 1)

        dg_t = Graph(node_feature=x, node_label=y, edge_index=edge_index, 
                     edge_feature=edge_x, edge_label=edge_y,
                     graph_feature=graph_x, graph_label=graph_y, directed=True)
        self.assertEqual(
            sorted(dg_t.keys),
            [
                'directed',
                'edge_feature',
                'edge_index',
                'edge_label',
                'edge_label_index',
                'graph_feature',
                'graph_label',
                'node_feature',
                'node_label',
                'node_label_index'
            ]
        )
        self.assertEqual(dg_t.num_nodes, G.number_of_nodes())
        self.assertEqual(dg_t.num_edges, G.number_of_edges())
        self.assertEqual(dg_t.num_node_features, 2)
        self.assertEqual(dg_t.num_edge_features, 2)
        self.assertEqual(dg_t.num_graph_features, 2)
        self.assertEqual(dg_t.num_node_labels, np.max(y.data.numpy()) + 1)
        self.assertEqual(dg_t.num_edge_labels, np.max(edge_y.data.numpy()) + 1)
        self.assertEqual(dg_t.num_graph_labels, np.max(graph_y.data.numpy()) + 1)

    def test_pyg_to_graph(self):
        pyg_dataset = Planetoid('./cora', 'Cora')

        dg = Graph.pyg_to_graph(pyg_dataset[0])
        pyg_data = pyg_dataset[0]
        self.assertEqual(pyg_data.num_nodes, dg.num_nodes)
        self.assertEqual(pyg_data.is_directed(), dg.is_directed())
        self.assertEqual(pyg_data.num_edges / 2, dg.num_edges)
        self.assertTrue(dg.num_node_features == pyg_data.x.shape[1])
        self.assertTrue(dg.num_node_labels == torch.max(pyg_data.y).item() + 1)
        self.assertTrue(dg.edge_index.shape == pyg_data.edge_index.shape)
        keys = [
            'G',
            'node_feature',
            'node_label',
            'edge_index',
            'edge_label_index',
            'node_label_index',
        ]
        self.assertTrue(tuple(dg.keys) == tuple(keys))

        # TODO: add the test for tensor backend after the change of the pyg_to_graph

    def test_clone(self):
        G, x, y, edge_x, edge_y, edge_index, graph_x, graph_y = (
            simple_networkx_graph()
        )
        Graph.add_edge_attr(G, "edge_feature", edge_x)
        Graph.add_edge_attr(G, "edge_label", edge_y)
        Graph.add_node_attr(G, "node_feature", x)
        Graph.add_node_attr(G, "node_label", y)
        Graph.add_graph_attr(G, "graph_feature", graph_x)
        Graph.add_graph_attr(G, "graph_label", graph_y)

        dg = Graph(G)
        dg1 = dg.clone()
        self.assertEqual(dg.num_nodes, dg1.num_nodes)
        self.assertEqual(dg.num_edges, dg1.num_edges)
        self.assertEqual(dg.num_node_features, dg1.num_node_features)
        self.assertEqual(dg.num_edge_features, dg1.num_edge_features)
        self.assertEqual(dg.num_node_labels, dg1.num_node_labels)
        self.assertEqual(dg.num_edge_labels, dg1.num_edge_labels)
        self.assertTrue(not id(dg.G) == id(dg1.G))
        self.assertTrue(not id(dg.edge_index) == id(dg1.edge_index))
        self.assertTrue(tuple(dg.keys) == tuple(dg1.keys))

        dg_t = Graph(node_feature=x, node_label=y, edge_index=edge_index, 
                     edge_feature=edge_x, edge_label=edge_y,
                     graph_feature=graph_x, graph_label=graph_y, directed=True)
        dg1_t = dg_t.clone()
        self.assertEqual(dg_t.num_nodes, dg1_t.num_nodes)
        self.assertEqual(dg_t.num_edges, dg1_t.num_edges)
        self.assertEqual(dg_t.num_node_features, dg1_t.num_node_features)
        self.assertEqual(dg_t.num_edge_features, dg1_t.num_edge_features)
        self.assertEqual(dg_t.num_node_labels, dg1_t.num_node_labels)
        self.assertEqual(dg_t.num_edge_labels, dg1_t.num_edge_labels)
        self.assertTrue(not id(dg_t.edge_index) == id(dg1_t.edge_index))
        self.assertTrue(tuple(dg_t.keys) == tuple(dg1_t.keys))

    def test_split_edge_case(self):
        G, x, y, edge_x, edge_y, edge_index, graph_x, graph_y = (
            simple_networkx_graph()
        )
        dg = Graph(G)

        dg_node = dg.split()
        dg_num_nodes_reduced = dg.num_nodes - 3
        self.assertEqual(
            dg_node[0].node_label_index.shape[0],
            1 + int(dg_num_nodes_reduced * 0.8),
        )
        self.assertEqual(
            dg_node[1].node_label_index.shape[0],
            1 + int(dg_num_nodes_reduced * 0.1),
        )
        self.assertEqual(
            dg_node[2].node_label_index.shape[0],
            dg.num_nodes
            - 2
            - int(dg_num_nodes_reduced * 0.8)
            - int(dg_num_nodes_reduced * 0.1)
        )

        dg_t = Graph(node_feature=x, edge_index=edge_index, directed=True)
        dg_node_t = dg.split()
        self.assertEqual(
            dg_node_t[0].node_label_index.shape[0],
            1 + int(dg_num_nodes_reduced * 0.8),
        )
        self.assertEqual(
            dg_node_t[1].node_label_index.shape[0],
            1 + int(dg_num_nodes_reduced * 0.1),
        )
        self.assertEqual(
            dg_node_t[2].node_label_index.shape[0],
            dg.num_nodes
            - 2
            - int(dg_num_nodes_reduced * 0.8)
            - int(dg_num_nodes_reduced * 0.1)
        )

        dg_edge = dg.split(task="edge")
        dg_num_edges_reduced = dg.num_edges - 3
        edge_0 = 1 + int(dg_num_edges_reduced * 0.8)
        edge_1 = 1 + int(dg_num_edges_reduced * 0.1)
        edge_2 = dg.num_edges - edge_0 - edge_1
        self.assertEqual(dg_edge[0].edge_label_index.shape[1], edge_0)
        self.assertEqual(dg_edge[1].edge_label_index.shape[1], edge_1)
        self.assertEqual(dg_edge[2].edge_label_index.shape[1], edge_2)

        dg_edge_t = dg_t.split(task="edge")
        self.assertEqual(dg_edge_t[0].edge_label_index.shape[1], edge_0)
        self.assertEqual(dg_edge_t[1].edge_label_index.shape[1], edge_1)
        self.assertEqual(dg_edge_t[2].edge_label_index.shape[1], edge_2)

        dg_link = dg.split(task="link_pred")
        dg_num_edges_reduced = dg.num_edges - 3
        edge_0 = 1 + int(dg_num_edges_reduced * 0.8)
        edge_1 = 1 + int(dg_num_edges_reduced * 0.1)
        edge_2 = dg.num_edges - edge_0 - edge_1
        self.assertEqual(dg_link[0].edge_label_index.shape[1], edge_0)
        self.assertEqual(dg_link[1].edge_label_index.shape[1], edge_1)
        self.assertEqual(dg_link[2].edge_label_index.shape[1], edge_2)

        dg_link_t = dg_t.split(task="link_pred")
        self.assertEqual(dg_link_t[0].edge_label_index.shape[1], edge_0)
        self.assertEqual(dg_link_t[1].edge_label_index.shape[1], edge_1)
        self.assertEqual(dg_link_t[2].edge_label_index.shape[1], edge_2)

    def test_split(self):
        pyg_dataset = Planetoid("./cora", "Cora")
        dg = Graph.pyg_to_graph(pyg_dataset[0])

        x = pyg_dataset[0].x
        y = pyg_dataset[0].y
        edge_index = pyg_dataset[0].edge_index
        dg_t = Graph(node_feature=x, node_label=y, edge_index=edge_index, directed=False)

        self.assertEqual(dg.is_directed(), dg_t.is_directed())

        dg_nodes = [dg.split(), dg_t.split()]
        dg_num_nodes_reduced = dg.num_nodes - 3
        for dg_node in dg_nodes:
            self.assertEqual(
                dg_node[0].node_label_index.shape[0],
                1 + int(dg_num_nodes_reduced * 0.8),
            )
            self.assertEqual(
                dg_node[1].node_label_index.shape[0],
                1 + int(dg_num_nodes_reduced * 0.1),
            )
            self.assertEqual(
                dg_node[2].node_label_index.shape[0],
                dg.num_nodes
                - 2
                - int(dg_num_nodes_reduced * 0.8)
                - int(dg_num_nodes_reduced * 0.1),
            )

        dg_edge = dg.split(task="edge")
        dg_num_edges_reduced = dg.num_edges - 3
        edge_0 = 2 * (1 + int(dg_num_edges_reduced * 0.8))
        edge_1 = 2 * (1 + int(dg_num_edges_reduced * 0.1))
        edge_2 = dg.num_edges * 2 - edge_0 - edge_1
        self.assertEqual(dg_edge[0].edge_label_index.shape[1], edge_0)
        self.assertEqual(dg_edge[1].edge_label_index.shape[1], edge_1)
        self.assertEqual(dg_edge[2].edge_label_index.shape[1], edge_2)

        dg_edge_t = dg_t.split(task="edge")
        self.assertEqual(dg_edge_t[0].edge_label_index.shape[1], edge_0)
        self.assertEqual(dg_edge_t[1].edge_label_index.shape[1], edge_1)
        self.assertEqual(dg_edge_t[2].edge_label_index.shape[1], edge_2)

        dg_link = dg.split(task="link_pred")
        dg_num_edges_reduced = dg.num_edges - 3
        edge_0 = 2 * (1 + int(dg_num_edges_reduced * 0.8))
        edge_1 = 2 * (1 + int(dg_num_edges_reduced * 0.1))
        edge_2 = dg.num_edges * 2 - edge_0 - edge_1
        self.assertEqual(dg_link[0].edge_label_index.shape[1], edge_0)
        self.assertEqual(dg_link[1].edge_label_index.shape[1], edge_1)
        self.assertEqual(dg_link[2].edge_label_index.shape[1], edge_2)

        dg_link_t = dg_t.split(task="link_pred")
        self.assertEqual(dg_link_t[0].edge_label_index.shape[1], edge_0)
        self.assertEqual(dg_link_t[1].edge_label_index.shape[1], edge_1)
        self.assertEqual(dg_link_t[2].edge_label_index.shape[1], edge_2)

        for message_ratio in [0.1, 0.2, 0.4, 0.8]:
            dg_link_resample = (
                dg_link[0].clone().resample_disjoint(
                    message_ratio=message_ratio,
                )
            )
            positive_edge_num = (
                int(0.5 * dg_link[0].clone().edge_label_index.shape[1])
            )
            self.assertEqual(
                dg_link_resample.edge_label_index.shape[1],
                2 * (
                    positive_edge_num
                    - 1
                    - int(message_ratio * (positive_edge_num - 2))
                )
            )

        for message_ratio in [0.1, 0.2, 0.4, 0.8]:
            dg_link_resample_t = (
                dg_link_t[0].clone().resample_disjoint(
                    message_ratio=message_ratio,
                )
            )
            positive_edge_num = (
                int(0.5 * dg_link_t[0].clone().edge_label_index.shape[1])
            )
            self.assertEqual(
                dg_link_resample_t.edge_label_index.shape[1],
                2 * (
                    positive_edge_num
                    - 1
                    - int(message_ratio * (positive_edge_num - 2))
                )
            )

        for split_ratio in [[0.1, 0.4, 0.5], [0.4, 0.3, 0.3], [0.7, 0.2, 0.1]]:
            dg_link_custom = (
                dg.split(task='link_pred', split_ratio=split_ratio)
            )
            dg_num_edges_reduced = dg.num_edges - 3
            edge_0 = 2 * (1 + int(dg_num_edges_reduced * split_ratio[0]))
            self.assertEqual(
                dg_link_custom[0].edge_label_index.shape[1],
                edge_0,
            )
            edge_1 = (
                2 * (
                    1
                    + int(split_ratio[0] * dg_num_edges_reduced)
                    + 1
                    + int(split_ratio[1] * dg_num_edges_reduced)
                )
                - edge_0
            )
            self.assertEqual(
                dg_link_custom[1].edge_label_index.shape[1],
                edge_1,
            )
            edge_2 = dg.num_edges * 2 - edge_0 - edge_1
            self.assertEqual(
                dg_link_custom[2].edge_label_index.shape[1],
                edge_2,
            )

        for split_ratio in [[0.1, 0.4, 0.5], [0.4, 0.3, 0.3], [0.7, 0.2, 0.1]]:
            dg_link_custom_t = (
                dg_t.split(task='link_pred', split_ratio=split_ratio)
            )
            dg_num_edges_reduced = dg.num_edges - 3
            edge_0 = 2 * (1 + int(dg_num_edges_reduced * split_ratio[0]))
            self.assertEqual(
                dg_link_custom_t[0].edge_label_index.shape[1],
                edge_0,
            )
            edge_1 = (
                2 * (
                    1
                    + int(split_ratio[0] * dg_num_edges_reduced)
                    + 1
                    + int(split_ratio[1] * dg_num_edges_reduced)
                )
                - edge_0
            )
            self.assertEqual(
                dg_link_custom_t[1].edge_label_index.shape[1],
                edge_1,
            )
            edge_2 = dg_t.num_edges * 2 - edge_0 - edge_1
            self.assertEqual(
                dg_link_custom_t[2].edge_label_index.shape[1],
                edge_2,
            )

    def test_transform(self):
        G, x, y, edge_x, edge_y, edge_index, graph_x, graph_y = (
            simple_networkx_graph()
        )
        Graph.add_edge_attr(G, "edge_feature", edge_x)
        Graph.add_edge_attr(G, "edge_label", edge_y)
        Graph.add_node_attr(G, "node_feature", x)
        Graph.add_node_attr(G, "node_label", y)
        Graph.add_graph_attr(G, "graph_feature", graph_x)
        Graph.add_graph_attr(G, "graph_label", graph_y)

        dg = Graph(G)
        dg_t = Graph(node_feature=x, node_label=y, edge_index=edge_index, 
                     edge_feature=edge_x, edge_label=edge_y,
                     graph_feature=graph_x, graph_label=graph_y, directed=True)
        dgs = [dg, dg_t]

        for dg in dgs:
            dg_edge_feature = dg.edge_feature.clone()
            dg_node_feature = dg.node_feature.clone()
            dg_graph_feature = dg.graph_feature.clone()

            dg.apply_tensor(
                lambda x: x, 'edge_feature', 'node_feature', 'graph_feature'
            )
            self.assertTrue(torch.all(dg_edge_feature.eq(dg.edge_feature)))
            self.assertTrue(torch.all(dg_node_feature.eq(dg.node_feature)))
            self.assertTrue(torch.all(dg_graph_feature.eq(dg.graph_feature)))

            dg.apply_tensor(
                lambda x: x + 10, 'edge_feature', 'node_feature', 'graph_feature'
            )
            self.assertFalse(torch.all(dg_edge_feature.eq(dg.edge_feature)))
            self.assertFalse(torch.all(dg_node_feature.eq(dg.node_feature)))
            self.assertFalse(torch.all(dg_graph_feature.eq(dg.graph_feature)))

            dg.apply_tensor(
                lambda x: x + 100, 'edge_feature', 'node_feature', 'graph_feature'
            )
            self.assertTrue(
                torch.all(dg.edge_feature.eq(dg_edge_feature + 10 + 100))
            )
            self.assertTrue(
                torch.all(dg.node_feature.eq(dg_node_feature + 10 + 100))
            )
            self.assertTrue(
                torch.all(dg.graph_feature.eq(dg_graph_feature + 10 + 100))
            )

            dg.apply_tensor(
                lambda x: x * 2, 'edge_feature', 'node_feature', 'graph_feature'
            )
            self.assertTrue(
                torch.all(dg.edge_feature.eq((dg_edge_feature + 10 + 100) * 2))
            )
            self.assertTrue(
                torch.all(dg.node_feature.eq((dg_node_feature + 10 + 100) * 2))
            )
            self.assertTrue(
                torch.all(dg.graph_feature.eq((dg_graph_feature + 10 + 100) * 2))
            )

    def test_repr(self):
        G, x, y, edge_x, edge_y, edge_index, graph_x, graph_y = (
            simple_networkx_graph()
        )
        Graph.add_edge_attr(G, "edge_feature", edge_x)
        Graph.add_edge_attr(G, "edge_label", edge_y)
        Graph.add_node_attr(G, "node_feature", x)
        Graph.add_node_attr(G, "node_label", y)
        Graph.add_graph_attr(G, "graph_feature", graph_x)
        Graph.add_graph_attr(G, "graph_label", graph_y)
        dg = Graph(G)
        dg_t = Graph(node_feature=x, node_label=y, edge_index=edge_index, 
                     edge_feature=edge_x, edge_label=edge_y,
                     graph_feature=graph_x, graph_label=graph_y, directed=True)
        self.assertEqual(
            repr(dg),
            "Graph(G=[], edge_feature=[17, 2], "
            "edge_index=[2, 17], edge_label=[17], edge_label_index=[2, 17], "
            "graph_feature=[1, 2], graph_label=[1], node_feature=[10, 2], "
            "node_label=[10], node_label_index=[10])"
        )
        self.assertEqual(
            repr(dg_t),
            "Graph(directed=[1], edge_feature=[17, 2], "
            "edge_index=[2, 17], edge_label=[17], edge_label_index=[2, 17], "
            "graph_feature=[1, 2], graph_label=[1], node_feature=[10, 2], "
            "node_label=[10], node_label_index=[10])"
        )


if __name__ == "__main__":
    unittest.main()
