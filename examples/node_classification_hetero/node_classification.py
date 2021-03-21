import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import (
    generate_convs,
    concatenate_citeseer_cora
)
from deepsnap.hetero_gnn import HeteroConv, HeteroSAGEConv, forward_op, loss_op
from torch_geometric.datasets import Planetoid
from deepsnap.dataset import GraphDataset
from deepsnap.batch import Batch
from deepsnap.hetero_graph import HeteroGraph
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: {}".format(device))
best_model = None
best_val = 0


class HeteroNet(torch.nn.Module):
    def __init__(self, hete, hidden_size, dropout):
        super(HeteroNet, self).__init__()
        conv1, conv2 = generate_convs(hete, HeteroSAGEConv, hidden_size)
        self.conv1 = HeteroConv(conv1)
        self.conv2 = HeteroConv(conv2)
        self.relus1 = nn.ModuleDict()
        self.relus2 = nn.ModuleDict()
        self.dropouts1 = nn.ModuleDict()
        self.dropouts2 = nn.ModuleDict()
        for node_type in hete.node_types:
            self.relus1[node_type] = nn.LeakyReLU()
            self.relus2[node_type] = nn.LeakyReLU()
            self.dropouts1[node_type] = nn.Dropout(p=dropout)
            self.dropouts2[node_type] = nn.Dropout(p=dropout)

    def forward(self, data):
        x = forward_op(data.node_feature, self.dropouts1)
        x = forward_op(x, self.relus1)
        x = self.conv1(x, data.edge_index)
        x = forward_op(x, self.dropouts2)
        x = forward_op(x, self.relus2)
        x = self.conv2(x, data.edge_index)
        return x

    def loss(self, pred, y, node_label_index):
        loss = loss_op(pred, y, node_label_index, F.cross_entropy)
        return loss


def train(model, optimizer, train_loader):
    model.train()
    optimizer.zero_grad()
    for batch in train_loader:
        batch.to(device)
        emb = model(batch)
        loss = model.loss(emb, batch.node_label, batch.node_label_index)
        loss.backward()
    optimizer.step()
    return loss.item()


def test(model, loaders):
    global best_model
    global best_val
    model.eval()
    accs = []
    for loader in loaders:
        for batch in loader:
            batch.to(device)
            logits = model(batch)
            total = 0
            acc = 0
            for node_type in logits:
                node_idx = batch.node_label_index[node_type].to(device)
                pred = logits[node_type][node_idx]
                pred = pred.max(1)[1]
                acc += pred.eq(
                    batch.node_label[node_type].to(device)
                ).sum().item()
                total += pred.size(0)
            acc /= total
            accs.append(acc)
    if accs[1] > best_val:
        best_val = accs[1]
        best_model = copy.deepcopy(model)
    return accs


if __name__ == "__main__":
    cora_pyg = Planetoid('./cora', 'Cora')
    citeseer_pyg = Planetoid('./citeseer', 'CiteSeer')
    G = concatenate_citeseer_cora(cora_pyg[0], citeseer_pyg[0])

    # The nodes in the graph have the features: node_feature, node_label and node_type ("cora_node" or "citeseer_node")
    print("The nodes in the concatenated heterogeneous graph have the following features:")
    for node in G.nodes(data=True):
        print(node[1])
        break
    # The edges in the graph have the features: edge_type ("cora_edge" or "citeseer_edge")
    print("The edges in the concatenated heterogeneous graph have the following features:")
    for edge in G.edges(data=True):
        print(edge[2])
        break

    hete = HeteroGraph(G)
    print(f"Heterogeneous graph {hete.num_nodes()} nodes, {hete.num_edges()} edges")

    dataset = GraphDataset([hete], task='node')
    dataset_train, dataset_val, dataset_test = dataset.split(
        transductive=True,
        split_ratio=[0.8, 0.1, 0.1]
    )
    train_loader = DataLoader(
        dataset_train, collate_fn=Batch.collate(), batch_size=16
    )
    val_loader = DataLoader(
        dataset_val, collate_fn=Batch.collate(), batch_size=16
    )
    test_loader = DataLoader(
        dataset_test, collate_fn=Batch.collate(), batch_size=16
    )
    loaders = [train_loader, val_loader, test_loader]

    hidden_size = 32
    model = HeteroNet(hete, hidden_size, 0.5).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.01, weight_decay=5e-3
    )
    num_epochs = 100

    train_accs, valid_accs, test_accs = [], [], []

    for epoch in range(num_epochs):
        loss = train(model, optimizer, train_loader)
        accs = test(model, loaders)
        log = "Epoch {}: Train: {:.4f}, Validation: {:.4f}. Test: {:.4f}"
        print(log.format(epoch + 1, accs[0], accs[1], accs[2]))
        train_accs.append(accs[0])
        valid_accs.append(accs[1])
        test_accs.append(accs[2])
    accs = test(best_model, loaders)
    log = "Best model: Train: {:.4f}, Validation: {:.4f}. Test: {:.4f}"
    print(log.format(accs[0], accs[1], accs[2]))
