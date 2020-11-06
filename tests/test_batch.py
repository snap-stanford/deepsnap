import math
import unittest
from copy import deepcopy
import torch
import networkx as nx
from torch.utils.data import DataLoader
from torch_geometric.datasets import TUDataset
from deepsnap.graph import Graph
from deepsnap.batch import Batch
from deepsnap.dataset import GraphDataset
from tests.utils import simple_networkx_graph


class TestBatch(unittest.TestCase):
    def test_batch_basic(self):
        G, x, y, edge_x, edge_y, edge_index, graph_x, graph_y = (
            simple_networkx_graph()
        )
        Graph.add_edge_attr(G, "edge_feature", edge_x)
        Graph.add_edge_attr(G, "edge_label", edge_y)
        Graph.add_node_attr(G, "node_feature", x)
        Graph.add_node_attr(G, "node_label", y)
        Graph.add_graph_attr(G, "graph_feature", graph_x)
        Graph.add_graph_attr(G, "graph_label", graph_y)
        H = deepcopy(G)
        graphs = [Graph(G), Graph(H)]
        batch = Batch.from_data_list(graphs)
        self.assertEqual(batch.num_graphs, 2)
        self.assertEqual(
            len(batch.node_feature),
            2 * len(graphs[0].node_feature),
        )

    def test_torch_dataloader_collate(self):
        # graph classification example
        pyg_dataset = TUDataset("./enzymes", "ENZYMES")
        graphs = GraphDataset.pyg_to_graphs(pyg_dataset)
        dataset = GraphDataset(graphs, task="graph")
        train_batch_num = math.ceil(len(dataset) * 0.8 / 32)
        test_batch_num = math.ceil(len(dataset) * 0.1 / 32)
        val_batch_num = math.ceil(len(dataset) * 0.1 / 32)
        datasets = {}
        datasets["train"], datasets["val"], datasets["test"] = (
            dataset.split(transductive=False, split_ratio=[0.8, 0.1, 0.1])
        )
        dataloaders = {
            split: DataLoader(
                dataset,
                collate_fn=Batch.collate(),
                batch_size=32,
                shuffle=True,
            )
            for split, dataset in datasets.items()
        }

        self.assertEqual(len(dataloaders["train"]), train_batch_num)
        self.assertEqual(len(dataloaders["val"]), test_batch_num)
        self.assertEqual(len(dataloaders["test"]), val_batch_num)
        for i, data in enumerate(dataloaders['train']):
            if i != len(dataloaders["train"]) - 1:
                self.assertEqual(data.num_graphs, 32)
        for i, data in enumerate(dataloaders["val"]):
            if i != len(dataloaders["val"]) - 1:
                self.assertEqual(data.num_graphs, 32)
        for i, data in enumerate(dataloaders["test"]):
            if i != len(dataloaders["test"]) - 1:
                self.assertEqual(data.num_graphs, 32)

    def test_collate_batch_nested(self):
        dims = [2, 3]
        G_sizes = [10, 5]
        G_list = []
        for i, size in enumerate(G_sizes):
            G = Graph()
            G.G = nx.complete_graph(i + 1)
            G.node_property = {
                "node_prop0": torch.ones(size, dims[0]) * i,
                "node_prop1": torch.ones(size, dims[1]) * i,
            }
            G_list.append(G)

        batch = Batch.from_data_list(G_list)
        self.assertEqual(batch.num_graphs, 2)
        self.assertEqual(
            batch.node_property["node_prop0"].size(0),
            sum(G_sizes)
        )

    def test_unbatch_nested(self):
        dims = [2, 3]
        G_sizes = [10, 5]
        G_list = []
        for i, size in enumerate(G_sizes):
            G = Graph()
            G.G = nx.complete_graph(i + 1)
            G.node_property = {
                "node_prop0": torch.ones(size, dims[0]) * i,
                "node_prop1": torch.ones(size, dims[1]) * i,
            }
            G_list.append(G)

        batch = Batch.from_data_list(G_list)

        # reconstruct graph list
        G_list_recon = batch.to_data_list()
        self.assertEqual(
            G_list_recon[0].node_property["node_prop0"].size(0),
            10,
        )
        self.assertEqual(
            G_list_recon[0].node_property["node_prop0"].size(1),
            2,
        )
        self.assertEqual(
            G_list_recon[1].node_property["node_prop1"].size(0),
            5,
        )
        self.assertEqual(
            G_list_recon[1].node_property["node_prop1"].size(1),
            3,
        )


if __name__ == "__main__":
    unittest.main()
