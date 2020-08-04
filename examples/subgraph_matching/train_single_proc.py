"""Train the order embedding model"""
import argparse
from collections import defaultdict
from itertools import permutations
import pickle
from queue import PriorityQueue
import os
import random
import time

from deepsnap.batch import Batch
import networkx as nx
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.metrics import precision_recall_curve
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
import torch_geometric.utils as pyg_utils
import torch_geometric.nn as pyg_nn

import data
#import data_random_basis as data
import models
import utils

#import matplotlib.pyplot as plt
#import matplotlib.colors as mcolors

def arg_parse():
    parser = argparse.ArgumentParser(description='GNN arguments.')

    parser.add_argument('--conv_type', type=str,
                        help='type of model')
    parser.add_argument('--method_type', type=str,
                        help='type of convolution')
    parser.add_argument('--batch_size', type=int,
                        help='Training batch size')
    parser.add_argument('--n_layers', type=int,
                        help='Number of graph conv layers')
    parser.add_argument('--hidden_dim', type=int,
                        help='Training hidden size')
    parser.add_argument('--max_graph_size', type=int,
                        help='max training graph size')
    parser.add_argument('--n_batches', type=int,
                        help='Number of training minibatches')
    parser.add_argument('--margin', type=float,
                        help='margin for loss')
    parser.add_argument('--dataset', type=str,
                        help='Dataset')
    parser.add_argument('--dataset_type', type=str,
                        help='"otf-syn" or "syn" or "real"')
    parser.add_argument('--eval_interval', type=int,
                        help='how often to eval during training')
    parser.add_argument('--val_size', type=int,
                        help='validation set size')
    parser.add_argument('--model_path', type=str,
                        help='path to save/load model')
    parser.add_argument('--start_weights', type=str,
                        help='file to load weights from')
    parser.add_argument('--test', action="store_true")
    parser.add_argument('--n_workers', type=int)

    parser.set_defaults(conv_type='SAGE',
                        method_type='order',
                        dataset='enzymes',
                        dataset_type='real',
                        n_layers=4,
                        batch_size=64,
                        hidden_dim=64,
                        dropout=0.0,
                        n_batches=1000000,
                        lr=1e-4,
                        margin=0.1,
                        test_set='',
                        eval_interval=100,
                        n_workers=4,
                        model_path="ckpt/model.pt",
                        start_weights='',
                        max_graph_size=20,
                        val_size=1024)

    return parser.parse_args()

def build_model(args):
    # build model

    # set the input dimension to be the dimension of node labels of the dataset
    if args.dataset == "enzymes":
        dim = 3
    elif args.dataset == "cox2":
        dim = 35
    elif args.dataset == "imdb-binary":
        dim = 1
    model = models.BaselineMLP(dim, args.hidden_dim, args)
    model.to(utils.get_device())
    if args.start_weights:
        model.load_state_dict(torch.load(args.start_weights,
            map_location=utils.get_device()))
    return model

def train_epoch(args, model, data_source, opt):
    """Train the order embedding model.

    args: Commandline arguments
    """
#    data_source = data.DataSource(dataset_name)
    batch_num = 0

    #for batch_num in range(args.n_batches):
    loaders = data_source.gen_data_loaders(args.batch_size, train=True)
    for batch_target, batch_neg_target, batch_neg_query in zip(*loaders):
        # train
        model.train()
        model.zero_grad()
        pos_a, pos_b, neg_a, neg_b = data_source.gen_batch(batch_target,
            batch_neg_target, batch_neg_query, True)
        pos_a = pos_a.to(utils.get_device())
        pos_b = pos_b.to(utils.get_device())
        neg_a = neg_a.to(utils.get_device())
        neg_b = neg_b.to(utils.get_device())
        emb_pos_a, emb_pos_b = model.emb_model(pos_a), model.emb_model(pos_b)
        emb_neg_a, emb_neg_b = model.emb_model(neg_a), model.emb_model(neg_b)
        emb_as = torch.cat((emb_pos_a, emb_neg_a), dim=0)
        emb_bs = torch.cat((emb_pos_b, emb_neg_b), dim=0)
        labels = torch.tensor([1]*pos_a.num_graphs + [0]*neg_a.num_graphs).to(
            utils.get_device())
        pred = model(emb_as, emb_bs)
        loss = model.criterion(pred, labels)
        loss.backward()
        if not args.test:
            opt.step()

        pred = model.predict(pred)
        train_acc = torch.mean((pred == labels).type(torch.float))
        train_loss = loss.item()
        print("Batch {}. Loss: {:.4f}. Training acc: {:.4f}".format(
            batch_num, train_loss, train_acc), end="               \r")
        batch_num += 1
        #logger.add_scalar("Loss/train", train_loss, 0)
        #logger.add_scalar("Accuracy/train", train_acc, 0)

def validation(args, model, data_source, logger, batch_n):
    # test on new motifs
    model.eval()
    all_raw_preds, all_preds, all_labels = [], [], []
    loaders = data_source.gen_data_loaders(args.batch_size, train=False)
    for batch_target, batch_neg_target, batch_neg_query in zip(*loaders):
        pos_a, pos_b, neg_a, neg_b = data_source.gen_batch(batch_target,
            batch_neg_target, batch_neg_query, False)
        pos_a = pos_a.to(utils.get_device())
        pos_b = pos_b.to(utils.get_device())
        neg_a = neg_a.to(utils.get_device())
        neg_b = neg_b.to(utils.get_device())
        with torch.no_grad():
            if args.dataset_type in ["real", "otf-syn"]:
                emb_pos_a, emb_pos_b = (model.emb_model(pos_a),
                    model.emb_model(pos_b))
                emb_neg_a, emb_neg_b = (model.emb_model(neg_a), 
                    model.emb_model(neg_b))
                emb_as = torch.cat((emb_pos_a, emb_neg_a), dim=0)
                emb_bs = torch.cat((emb_pos_b, emb_neg_b), dim=0)
                labels = torch.tensor([1]*pos_a.num_graphs +
                    [0]*neg_a.num_graphs).to(utils.get_device())
            raw_pred = model(emb_as, emb_bs)
            pred = model.predict(raw_pred)
            raw_pred = raw_pred[:,1]
        all_raw_preds.append(raw_pred)
        all_preds.append(pred)
        all_labels.append(labels)
    pred = torch.cat(all_preds, dim=-1)
    labels = torch.cat(all_labels, dim=-1)
    raw_pred = torch.cat(all_raw_preds, dim=-1)
    acc = torch.mean((pred == labels).type(torch.float))
    prec = (torch.sum(pred * labels).item() / torch.sum(pred).item() if
        torch.sum(pred) > 0 else float("NaN"))
    recall = (torch.sum(pred * labels).item() /
        torch.sum(labels).item() if torch.sum(labels) > 0 else
        float("NaN"))
    labels = labels.detach().cpu().numpy()
    raw_pred = raw_pred.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()
    auroc = roc_auc_score(labels, raw_pred)
    tn, fp, fn, tp = confusion_matrix(labels, pred).ravel()

    print("\nValidation. Acc: {:.4f}. "
        "P: {:.4f}. R: {:.4f}. AUROC: {:.4f}\n     "
        "TN: {}. FP: {}. FN: {}. TP: {}".format(
            acc, prec, recall, auroc,
            tn, fp, fn, tp))
    logger.add_scalar("Accuracy/test", acc, batch_n)
    logger.add_scalar("Precision/test", prec, batch_n)
    logger.add_scalar("Recall/test", recall, batch_n)
    logger.add_scalar("AUROC/test", auroc, batch_n)
    logger.add_scalar("TP/test", tp, batch_n)
    logger.add_scalar("TN/test", tn, batch_n)
    logger.add_scalar("FP/test", fp, batch_n)
    logger.add_scalar("FN/test", fn, batch_n)

def main():
    args = arg_parse()
    # see test-tube
    #args = hyp_search.hyp_arg_parse()

    if not os.path.exists(os.path.dirname(args.model_path)):
        os.makedirs(os.path.dirname(args.model_path))

    print("Starting {} workers".format(args.n_workers))
    print("Using dataset {}".format(args.dataset))

    record_keys = ["conv_type", "n_layers", "hidden_dim",
        "margin", "dataset", "dataset_type", "max_graph_size", "skip"]
    args_str = ".".join(["{}={}".format(k, v)
        for k, v in sorted(vars(args).items()) if k in record_keys])
    logger = SummaryWriter("log/" + args_str)

    model = build_model(args)

    data_source = data.DataSource(args.dataset)
    opt = optim.Adam(model.parameters(), args.lr)

    if args.test:
        validation(args, model, data_source, logger, 0, make_pr_curve=True)
    else:
        batch_n = 0
        for epoch in range(args.n_batches // args.eval_interval):
            print("Epoch", epoch)
            train_epoch(args, model, data_source, opt)
            validation(args, model, data_source, logger, batch_n)
            if not args.test:
                print("Saving {}".format(args.model_path))
                torch.save(model.state_dict(), args.model_path)

if __name__ == '__main__':
    main()
