# need to add neighbor sampling, API calls to replace torch.distributed

import math
import copy
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import SplineConv
from torch_geometric.nn import GCNConv
import sys
import networkx as nx
import pdb

from deepsnap.dataset import GraphDataset
from deepsnap.batch import Batch
from torch.utils.data import DataLoader

from torch.nn.parallel import DistributedDataParallel

n_gpus = 2
name = 'Cora'
model_name = 'GCN'
fixed_split = True
pyg_dataset = Planetoid('./cora', name,
                        transform=T.TargetIndegree())  # load some format of graph data

if not fixed_split:
    graphs = GraphDataset.pyg_to_graphs(pyg_dataset, verbose=True, fixed_split=fixed_split)  # transform to our format

    dataset = GraphDataset(graphs, task='node')  # node, edge, link_pred, graph
    dataset_train, dataset_val, dataset_test = dataset.split(
        transductive=True,
        split_ratio=[0.8, 0.1, 0.1])  # transductive split, inductive split

else:
    graphs_train, graphs_val, graphs_test = \
        GraphDataset.pyg_to_graphs(pyg_dataset, verbose=True, fixed_split=fixed_split)  # transform to our format

    dataset_train, dataset_val, dataset_test = \
        GraphDataset(graphs_train, task='node'), GraphDataset(graphs_val, task='node'), GraphDataset(graphs_test, task='node')
                         
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        if model_name == 'GCN':
            self.conv1 = GCNConv(dataset_train.num_node_features, 16)
            self.conv2 = GCNConv(16, dataset_train.num_node_labels)
        elif model_name == 'Spline':
            self.conv1 = SplineConv(dataset_train.num_node_features, 16, dim=1, kernel_size=2)
            self.conv2 = SplineConv(16, dataset_train.num_node_labels, dim=1, kernel_size=2)

    def forward(self, batch):
        x, edge_index, edge_feature = \
            batch.node_feature, batch.edge_index, batch.edge_feature
        x = F.dropout(x, training=self.training)
        if model_name == 'GCN':
            x = F.elu(self.conv1(x, edge_index))
        elif model_name == 'Spline':
            x = F.elu(self.conv1(x, edge_index, edge_feature))
        x = F.dropout(x, training=self.training)
        if model_name == 'GCN':
            x = self.conv2(x, edge_index)
        elif model_name == 'Spline':
            x = self.conv2(x, edge_index, edge_feature)
        return F.log_softmax(x, dim=1)

def train(proc_id):
    model.train()
    optimizer.zero_grad()
    for batch in train_loader:
        batch.to(device)
        emb = model(batch)
        loss = F.nll_loss(emb[batch.node_label_index],
                          batch.node_label[batch.node_label_index])
        loss.backward()
        
        if n_gpus > 1:
            torch.distributed.barrier()
            for param in model.parameters():
                if param.requires_grad and param.grad is not None:
                    torch.distributed.all_reduce(param.grad.data, op=torch.distributed.ReduceOp.SUM)
                    param.grad.data /= n_gpus
            torch.distributed.barrier()
        optimizer.step()


def test():
    model.eval()
    accs = []
    for loader in [train_loader, val_loader, test_loader]:
        for batch in loader:
            batch.to(device)
            logits = model(batch)
            pred = logits[batch.node_label_index].max(1)[1]
            acc = pred.eq(batch.node_label[batch.node_label_index]).sum().item()
            total = batch.node_label_index.shape[0]
            acc /= total
            accs.append(acc)
    return accs

def run(proc_id, n_gpus, devices):
    dataset_train = torch.split(dataset_train, len(dataset_train) // n_gpus)[proc_id]

    train_loader = DataLoader(dataset_train, collate_fn=Batch.collate(),
                          batch_size=16)  # basic data loader
    val_loader = DataLoader(dataset_val, collate_fn=Batch.collate(),
                          batch_size=16)  # basic data loader
    test_loader = DataLoader(dataset_test, collate_fn=Batch.collate(),
                          batch_size=16)  # basic data loader
    
    dev_id = devices[proc_id]
    torch.cuda.set_device(dev_id)
    model = Net().to(dev_id)
    model.reset_parameters()
    if n_gpus > 1:
        model = DistributedDataParallel(model, device_ids=[dev_id], output_device=dev_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-3)
    
    val_max = -math.inf
    best_model = model
    if n_gpus > 1:
            dist_init_method = 'tcp://{master_ip}:{master_port}'.format(master_ip='127.0.0.1', master_port='12346')
            world_size = n_gpus
            torch.distributed.init_process_group(backend="nccl", init_method=dist_init_method, world_size=world_size,rank=proc_id)
    for epoch in range(1, 201):
        train()
        log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        train_acc, val_acc, test_acc = test()
        print(log.format(epoch, train_acc, val_acc, test_acc))
        if val_max < val_acc:
            val_max = val_acc
            # best_model = copy.deepcopy(model)

    # model = best_model
    log = 'Best, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    train_acc, val_acc, test_acc = test()
    print(log.format(train_acc, val_acc, test_acc))


if n_gpus == 1:
    run(0, n_gpus, devices)
else:
    mp.spawn(run, args=(n_gpus, devices), nprocs=n_gpus, join=True)
