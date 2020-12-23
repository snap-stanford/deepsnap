import snap
import snapx as sx
import unittest
import torch
import networkx as nx
from torch.utils.data import DataLoader
from torch_geometric.datasets import Planetoid
from deepsnap.graph import Graph
from deepsnap.batch import Batch
from deepsnap.dataset import GraphDataset


class TestGraphBackend(unittest.TestCase):

    def test_swith_graph_backend_global(self):
        import deepsnap
        self.assertTrue(isinstance(deepsnap._netlib, type(sx)))
        import networkx as nx
        deepsnap.use(nx)
        self.assertTrue(isinstance(deepsnap._netlib, type(nx)))
        deepsnap.use(sx)
        self.assertTrue(isinstance(deepsnap._netlib, type(sx)))

    def test_specify_graph_backend_init(self):
        G = sx.Graph()
        G.add_nodes_from(range(100))
        G.add_edges_from([[0, 4], [1, 5], [2, 6]])
        graph = Graph(G, netlib=sx)
        self.assertTrue(isinstance(graph.G, sx.Graph))
        self.assertEqual(list(graph.edge_index.shape), [2, 6])
        self.assertEqual(list(graph.edge_label_index.shape), [2, 6])
        self.assertEqual(list(graph.node_label_index.shape), [100])

        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from(range(100))
        G.add_edges_from([[0, 4], [1, 5], [2, 6]])
        graph = Graph(G, netlib=nx)
        self.assertTrue(isinstance(graph.G, nx.Graph))
        self.assertEqual(list(graph.edge_index.shape), [2, 6])
        self.assertEqual(list(graph.edge_label_index.shape), [2, 6])
        self.assertEqual(list(graph.node_label_index.shape), [100])

    def test_pyg_to_graph_global(self):
        import deepsnap
        deepsnap.use(nx)

        pyg_dataset = Planetoid('./planetoid', "Cora")
        pyg_data = pyg_dataset[0]
        graph = Graph.pyg_to_graph(pyg_data)
        self.assertTrue(isinstance(graph.G, nx.Graph))

        deepsnap.use(sx)
        graph = Graph.pyg_to_graph(pyg_data)
        self.assertTrue(isinstance(graph.G, sx.Graph))

    def test_pyg_to_graphs_global(self):
        import deepsnap
        deepsnap.use(nx)

        pyg_dataset = Planetoid('./planetoid', "Cora")
        graphs = GraphDataset.pyg_to_graphs(pyg_dataset)
        self.assertTrue(isinstance(graphs[0].G, nx.Graph))
        dataset = GraphDataset(graphs, task='node')
        num_nodes = dataset.num_nodes[0]
        node_0 = int(0.8 * num_nodes)
        node_1 = int(0.1 * num_nodes)
        node_2 = num_nodes - node_0 - node_1
        train, val, test = dataset.split()
        self.assertTrue(isinstance(train[0].G, nx.Graph))
        self.assertTrue(isinstance(val[0].G, nx.Graph))
        self.assertTrue(isinstance(test[0].G, nx.Graph))
        self.assertEqual(train[0].node_label_index.shape[0], node_0)
        self.assertEqual(val[0].node_label_index.shape[0], node_1)
        self.assertEqual(test[0].node_label_index.shape[0], node_2)

        train_loader = DataLoader(
            train, collate_fn=Batch.collate(), batch_size=1
        )
        for batch in train_loader:
            self.assertTrue(isinstance(batch.G[0], nx.Graph))

        deepsnap.use(sx)
        graphs = GraphDataset.pyg_to_graphs(pyg_dataset)
        self.assertTrue(isinstance(graphs[0].G, sx.Graph))
        dataset = GraphDataset(graphs, task='node')
        num_nodes = dataset.num_nodes[0]
        node_0 = int(0.8 * num_nodes)
        node_1 = int(0.1 * num_nodes)
        node_2 = num_nodes - node_0 - node_1
        train, val, test = dataset.split()
        self.assertTrue(isinstance(train[0].G, sx.Graph))
        self.assertTrue(isinstance(val[0].G, sx.classes.graph.Graph))
        self.assertTrue(isinstance(test[0].G, sx.classes.graph.Graph))
        self.assertEqual(train[0].node_label_index.shape[0], node_0)
        self.assertEqual(val[0].node_label_index.shape[0], node_1)
        self.assertEqual(test[0].node_label_index.shape[0], node_2)

        train_loader = DataLoader(
            train, collate_fn=Batch.collate(), batch_size=1
        )
        for batch in train_loader:
            self.assertTrue(isinstance(batch.G[0], sx.Graph))

    def test_pyg_to_graphs_init(self):
        pyg_dataset = Planetoid('./planetoid', "Cora")
        graphs = GraphDataset.pyg_to_graphs(pyg_dataset, netlib=nx)
        self.assertTrue(isinstance(graphs[0].G, nx.Graph))
        dataset = GraphDataset(graphs, task='node')
        num_nodes = dataset.num_nodes[0]
        node_0 = int(0.8 * num_nodes)
        node_1 = int(0.1 * num_nodes)
        node_2 = num_nodes - node_0 - node_1
        train, val, test = dataset.split()
        self.assertTrue(isinstance(train[0].G, nx.Graph))
        self.assertTrue(isinstance(val[0].G, nx.Graph))
        self.assertTrue(isinstance(test[0].G, nx.Graph))
        self.assertEqual(train[0].node_label_index.shape[0], node_0)
        self.assertEqual(val[0].node_label_index.shape[0], node_1)
        self.assertEqual(test[0].node_label_index.shape[0], node_2)
        train_loader = DataLoader(
            train, collate_fn=Batch.collate(), batch_size=1
        )
        for batch in train_loader:
            self.assertTrue(isinstance(batch.G[0], nx.Graph))

        graphs = GraphDataset.pyg_to_graphs(pyg_dataset, netlib=sx)
        self.assertTrue(isinstance(graphs[0].G, sx.Graph))
        dataset = GraphDataset(graphs, task='node')
        num_nodes = dataset.num_nodes[0]
        node_0 = int(0.8 * num_nodes)
        node_1 = int(0.1 * num_nodes)
        node_2 = num_nodes - node_0 - node_1
        train, val, test = dataset.split()
        self.assertTrue(isinstance(train[0].G, sx.classes.graph.Graph))
        self.assertTrue(isinstance(val[0].G, sx.classes.graph.Graph))
        self.assertTrue(isinstance(test[0].G, sx.classes.graph.Graph))
        self.assertEqual(train[0].node_label_index.shape[0], node_0)
        self.assertEqual(val[0].node_label_index.shape[0], node_1)
        self.assertEqual(test[0].node_label_index.shape[0], node_2)
        train_loader = DataLoader(
            train, collate_fn=Batch.collate(), batch_size=1
        )
        for batch in train_loader:
            self.assertTrue(isinstance(batch.G[0], sx.Graph))


if __name__ == "__main__":
    unittest.main()
