import sys
sys.path.append('../../')

import random
import os

import numpy as np
import torch_geometric
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch_geometric.utils import from_networkx
from nxdataset import NxDataset
import networkx as nx
from tensorboardX import SummaryWriter

from configs import arg_parse
import gengraph
from models_pyg import GCNNet
import utils.io_utils as io_utils
from utils import featgen
import utils.train_utils as train_utils

def test(loader, model, args, labels, test_mask):
    model.eval()

    train_ratio = args.train_ratio
    correct, total = 0, 0
    for data in loader:
        with torch.no_grad():
            pred = model(data)
            pred = pred.argmax(dim=1)

        pred = pred[test_mask]
        label = labels[test_mask]
            
        correct += pred.eq(label).sum().item()
        total += len(pred)
    
    return correct / total

def syn_task1(args, writer=None):
    print ('Generating graph.')
    feature_generator=featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))
    if args.dataset == 'syn1':
        gen_fn = gengraph.gen_syn1
    elif args.dataset == 'syn2':
        gen_fn = gengraph.gen_syn2
        feature_generator = None
    elif args.dataset == 'syn3':
        gen_fn = gengraph.gen_syn3
    elif args.dataset == 'syn4':
        gen_fn = gengraph.gen_syn4
    elif args.dataset == 'syn5':
        gen_fn = gengraph.gen_syn5
    G, labels, name = gen_fn(feature_generator=feature_generator)
    pyg_G = NxDataset([G], device=torch.device('gpu' if args.gpu else 'cpu'))[0]
    num_classes = max(labels)+1
    labels = torch.LongTensor(labels)
    print ('Done generating graph.')

    model = GCNNet(args.input_dim, args.hidden_dim, args.output_dim, num_classes, args.num_gc_layers, args=args)
    
    if args.gpu:
        model = model.cuda()

    train_ratio = args.train_ratio
    num_train = int(train_ratio * G.number_of_nodes())
    num_test = G.number_of_nodes() - num_train

    idx = [i for i in range(G.number_of_nodes())]

    np.random.shuffle(idx)
    train_mask = idx[:num_train]
    test_mask = idx[num_train:]

    loader = torch_geometric.data.DataLoader([pyg_G], batch_size=1)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler, opt = train_utils.build_optimizer(args, model.parameters(),
            weight_decay=args.weight_decay)
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        for batch in loader:
            opt.zero_grad()
            pred = model(batch)
    
            pred = pred[train_mask]
            label = labels[train_mask]
            loss = model.loss(pred, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            total_loss += loss.item() * 1
        writer.add_scalar("loss", total_loss, epoch)
        
        if epoch % 10 == 0:
            test_acc = test(loader, model, args, labels, test_mask)
            print("{} {:.4f} {:.4f}".format(epoch, total_loss, test_acc))
            writer.add_scalar("test", test_acc, epoch)

    print("{} {:.4f} {:.4f}".format(epoch, total_loss, test_acc))
    data = gengraph.preprocess_input_graph(G, labels)
    adj = torch.tensor(data['adj'], dtype=torch.float)
    x = torch.tensor(data['feat'], requires_grad=True, dtype=torch.float)

    model.eval()
    ypred = model(batch)

prog_args = arg_parse()
path = os.path.join(prog_args.logdir, io_utils.gen_prefix(prog_args) + '_pyg')
syn_task1(prog_args, writer=SummaryWriter(path))
