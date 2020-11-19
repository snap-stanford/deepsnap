import os
import torch
import argparse
import skip_models
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
import torch_geometric.transforms as T
import torch_geometric.nn as pyg_nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch_geometric.datasets import TUDataset
from sklearn.metrics import *
from torch.nn import Sequential, Linear, ReLU
from deepsnap.dataset import GraphDataset
from deepsnap.batch import Batch
from transforms import *

# use for graph or node classification where you want to automatically split batches to each worker
def model_parallelize_with_split(dset, model, rank, num_workers = None, addr = 'localhost', prt = '12355', backend = 'nccl'):
    dset[0] = dset[0].split(dset[0].size(0) // world_size)[rank]
    return model_parallelize(model,rank,num_workers,addr,prt,backend), dset

# use when you want to deal with splitting data among workers yourself, (i.e. for link prediction)
def model_parallelize(model, rank, num_workers = None, addr = 'localhost', prt = '12355', backend = 'nccl', single_node=False):
    if num_workers != None:
        world_size = torch.cuda.device_count()
    else:
        world_size = num_workers
    os.environ['MASTER_ADDR'] = addr
    os.environ['MASTER_PORT'] = prt
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    if single_node:
        model = DistributedDataParallel(model,device_ids=[rank])
    else: 
        model = DistributedDataParallel(model,
                                        device_ids=[args.local_rank],
                                        output_device=args.local_rank)
    return model
    
# use to overwrite default gradient synchronizer with your own reduce operation
def parallel_sync(model, syncop = torch.distributed.ReduceOp.SUM, divide_by_n = True, n_gpus = 2):
    for param in model.parameters():
        if param.requires_grad and param.grad is not None:
            torch.distributed.all_reduce(param.grad.data, op=syncop)
            if divide_by_n:
                param.grad.data /= n_gpus

def arg_parse():
    parser = argparse.ArgumentParser(description='Graph classification arguments.')

    parser.add_argument('--device', type=str,
                        help='CPU / GPU device.')
    parser.add_argument('--epochs', type=int,
                        help='Number of epochs to train.')
    parser.add_argument('--dataset', type=str,
                        help='Graph classification dataset. dd or enzymes.')
    parser.add_argument('--model', type=str,
                        help='GCN, GAT, GraphSAGE or GIN.')
    parser.add_argument('--batch_size', type=int,
                        help='Batch size for graph classification.')
    parser.add_argument('--hidden_dim', type=int,
                        help='Hidden dimension of GNN.')
    parser.add_argument('--num_layers', type=int,
                        help='Number of graph convolution layers.')
    parser.add_argument('--opt', type=str,
                        help='Optimizer such as adam, sgd, rmsprop or adagrad.')
    parser.add_argument('--weight_decay', type=float,
                        help='Weight decay.')
    parser.add_argument('--dropout', type=float,
                        help='The dropout ratio.')
    parser.add_argument('--lr', type=float,
                        help='Learning rate.')
    parser.add_argument('--skip', type=str,
                        help='Skip connections for GCN, GAT or GraphSAGE if specified as last.')
    parser.add_argument('--transform_dataset', type=str,
                        help='apply transform to the whole dataset.')
    parser.add_argument('--transform_batch', type=str,
                        help='apply transform to each batch.')
    parser.add_argument('--radius', type=int,
                        help='Radius of mini-batch ego networks')
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--single_node', type=bool)
    parser.add_argument('--world_size', type=int)

    parser.set_defaults(
            device='cuda:0', 
            epochs=500,
            dataset='enzymes',
            model='GIN',
            batch_size=32,
            hidden_dim=20,
            num_layers=3,
            opt='adam',
            weight_decay=5e-4,
            dropout=0.2,
            lr=0.001,
            skip=None,
            transform_dataset=None,
            transform_batch=None,
            radius=3,
            single_node=False,
            world_size=2
    )
    return parser.parse_args()


def get_transform(name):
    transform_funcs = {
        'ego': ego_nets,
        'path': path_len,
    }
    assert name in transform_funcs.keys(), \
        'Transform function \'{}\' not supported'.format(name)
    return transform_funcs[name]


def build_optimizer(args, params):
    weight_decay = args.weight_decay
    filter_fn = filter(lambda p : p.requires_grad, params)
    if args.opt == 'adam':
        optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter_fn, lr=args.lr, momentum=0.95, weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'adagrad':
        optimizer = optim.Adagrad(filter_fn, lr=args.lr, weight_decay=weight_decay)
    return optimizer

class GIN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, args):
        super(GIN, self).__init__()
        self.num_layers = args.num_layers
        
        self.pre_mp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim))

        self.convs = nn.ModuleList()
        self.nn1 = Sequential(Linear(hidden_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
        self.convs.append(pyg_nn.GINConv(self.nn1))
        for l in range(args.num_layers-1):
            self.nnk = Sequential(Linear(hidden_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
            self.convs.append(pyg_nn.GINConv(self.nnk))

        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), ReLU(), nn.Linear(hidden_dim, output_dim))

    def forward(self, data):
        x, edge_index, batch = data.node_feature, data.edge_index, data.batch
        x = self.pre_mp(x)
        for i in range(len(self.convs)):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
        x = pyg_nn.global_add_pool(x, batch)
        x = self.post_mp(x)
        x = F.log_softmax(x, dim=1)
        return x

    def loss(self, pred, label):
        return F.nll_loss(pred, label)

class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, args):
        super(GNN, self).__init__()
        self.dropout = args.dropout
        self.num_layers = args.num_layers

        conv_model = self.build_conv_model(args.model)
        self.convs = nn.ModuleList()
        self.convs.append(conv_model(input_dim, hidden_dim))

        for l in range(args.num_layers-1):
            self.convs.append(conv_model(hidden_dim, hidden_dim))

        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Dropout(args.dropout), 
            ReLU(),
            nn.Linear(hidden_dim, output_dim))

    def build_conv_model(self, model_type):
        if model_type == 'GCN':
            return pyg_nn.GCNConv
        elif model_type == 'GAT':
       	    return pyg_nn.GATConv
        elif model_type == "GraphSage":
            return pyg_nn.SAGEConv
        else:
            raise ValueError("Model {} unavailable, please add it to GNN.build_conv_model.".format(model_type))

    def forward(self, data):
        x, edge_index, batch = data.node_feature, data.edge_index, data.batch

        for i in range(len(self.convs)):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # x = pyg_nn.global_mean_pool(x, batch)
        x = pyg_nn.global_add_pool(x, batch)
        x = self.post_mp(x)
        x = F.log_softmax(x, dim=1)
        return x

    def loss(self, pred, label):
        return F.nll_loss(pred, label)

def train(model, train_loader, val_loader, test_loader, args, num_node_features, num_classes, device="0"):
    
    opt = build_optimizer(args, model.parameters())

    for epoch in range(args.epochs):
        total_loss = 0
        model.train()
        num_graphs = 0
        for batch in train_loader:
            if args.transform_batch is not None:
                trans_func = get_transform(args.transform_batch)
                batch.apply_transform(trans_func, radius=args.radius)
            batch.to(device)
            opt.zero_grad()
            pred = model(batch)
            label = batch.graph_label
            loss = F.nll_loss(pred, label)
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch.num_graphs
            num_graphs += batch.num_graphs
        total_loss /= num_graphs

        train_acc = test(train_loader, model, args, device)
        val_acc = test(val_loader, model, args, device)
        test_acc = test(test_loader, model, args, device)
        print("Epoch {}: Train: {:.4f}, Validation: {:.4f}. Test: {:.4f}, Loss: {:.4f}".format(epoch + 1, train_acc, val_acc, test_acc, total_loss))

def test(loader, model, args, device='cuda'):
    model.eval()

    correct = 0
    num_graphs = 0
    for batch in loader:
        if args.transform_batch is not None:
            trans_func = get_transform(args.transform_batch)
            batch.apply_transform(trans_func, radius=args.radius)
        batch.to(device)
        with torch.no_grad():
            pred = model(batch).max(dim=1)[1]
            label = batch.graph_label
        correct += pred.eq(label).sum().item()
        num_graphs += batch.num_graphs
    # print("loader len {}".format(num_graphs))
    return correct / num_graphs

def run(rank, world_size, args, single_node = True):
    if single_node:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group('nccl', rank=rank, world_size=world_size)
    if args.dataset == 'enzymes':
        pyg_dataset = TUDataset('./enzymes', 'ENZYMES')
    elif args.dataset == 'dd':
        pyg_dataset = TUDataset('./dd', 'DD')
    else:
        raise ValueError("Unsupported dataset.")

    graphs = GraphDataset.pyg_to_graphs(pyg_dataset)

    dataset = GraphDataset(graphs, task="graph")
    datasets = {}
    datasets['train'], datasets['val'], datasets['test'] = dataset.split(
            transductive=False, split_ratio = [0.8, 0.1, 0.1])

    if args.transform_dataset is not None:
        trans_func = get_transform(args.transform_dataset)
        for _, dataset in datasets.items():
            dataset.apply_transform(trans_func, radius=args.radius)

    dataloaders = {split: DataLoader(
                dataset, collate_fn=Batch.collate(), 
                batch_size=args.batch_size, shuffle=True)
                for split, dataset in datasets.items()}

    num_classes = datasets['train'].num_graph_labels
    num_node_features = datasets['train'].num_node_features

    if args.skip is not None:
        model_cls = skip_models.SkipLastGNN
    elif args.model == "GIN":
        model_cls = GIN
    else:
        model_cls = GNN

    model = model_cls(num_node_features, args.hidden_dim, num_classes, args).to(rank)
    if single_node:
        model = DistributedDataParallel(model, device_ids=[rank])
    else:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                  device_ids=[args.local_rank],
                                                  output_device=args.local_rank)

    train(model, dataloaders['train'], dataloaders['val'], dataloaders['test'], 
            args, num_node_features, num_classes, rank)
    dist.destroy_process_group()

if __name__ == "__main__":
    args = arg_parse()
    single_node = args.single_node
    if single_node:
        world_size = args.world_size
        print('Let\'s use', world_size, 'GPUs!')
        mp.spawn(run, args=(world_size, args), nprocs=world_size, join=True)
    else:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group('nccl', rank=rank)
        
        # local rank represents worker ID within one node
        local_rank = args.local_rank # local rank is useful for assigning data to GPU #
        torch.cuda.set_device(local_rank)
        # global rank represents worker ID across all nodes
        global_rank = dist.get_rank() # global rank is useful for data splitting 
        # world_size is set by torch.distributed.launch in this case
        world_size = torch.distributed.get_world_size()
        
        run(local_rank, world_size,args, single_node = False)

'''
#import horovod.torch as hvd
# (INCOMPLETE) use when you would like to use horovod to parallelize
def model_parallelize_horovod(kwargs):
    hvd.init()
    torch.manual_seed(args.seed)
    if args.cuda:
        # Horovod: pin GPU to local rank.
        torch.cuda.set_device(hvd.local_rank())
        torch.cuda.manual_seed(args.seed)
    torch.set_num_threads(1)
    if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
            mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
        kwargs['multiprocessing_context'] = 'forkserver'
    # --- Need to replace sampling/data loading with equivalent DeepSNAP code ---
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler, **kwargs)
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size,
                                              sampler=test_sampler, **kwargs)
    # --- end code to replace ---
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)
    compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none
    optimizer = hvd.DistributedOptimizer(optimizer,
                                         named_parameters=model.named_parameters(),
                                         compression=compression,
                                         op=hvd.Adasum if args.use_adasum else hvd.Average,
                                         gradient_predivide_factor=args.gradient_predivide_factor)
    return model, optimizer, train_loader, test_loader
'''
