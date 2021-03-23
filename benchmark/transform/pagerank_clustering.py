import time
import torch
import argparse
import deepsnap
import numpy as np

from deepsnap.dataset import GraphDataset
from torch_geometric.datasets import TUDataset

def arg_parse():
    parser = argparse.ArgumentParser(description='Pagerank arguments.')
    parser.add_argument('--device', type=str,
                        help='CPU / GPU device.')
    parser.add_argument('--netlib', type=str,
                        help='Backend network library, nx or sx.')
    parser.add_argument('--batch_size', type=int,
                        help='Batch size.')
    parser.add_argument('--num_runs', type=int,
                        help='Number of runs averaged on.')
    parser.add_argument('--dataset', type=str,
                        help='Dataset.')
    parser.add_argument('--print_run', action='store_true',
                        help='Print out current run.')
    parser.add_argument('--bench_task', type=str,
                        help='Either page or cluster.')

    parser.set_defaults(
        device='cuda:0',
        netlib="nx",
        batch_size=1,
        num_runs=100,
        dataset='COX2',
        print_run=False,
        bench_task='page',
    )
    return parser.parse_args()

################################################################################
# DeepSNAP transformation function
def page_fun(graph, lib):
    if lib == "nx":
        pages = netlib.pagerank(graph.G)
    elif lib == "sx":
        pages = graph.G._graph.GetPageRank()
    feature = torch.tensor([pages[node] for node in range(graph.num_nodes)], dtype=torch.float32).view(graph.num_nodes, 1)
    graph.node_feature = torch.cat([graph.node_feature, feature], dim=-1)

def clustering_fun(graph, lib):
    if lib == "nx":
        coefficients = netlib.clustering(graph.G)
    else:
        coefficients = {}
        for node in range(graph.num_nodes):
            coefficients[node] = graph.G._graph.GetNodeClustCf(node)
    feature = torch.tensor([coefficients[node] for node in range(graph.num_nodes)], dtype=torch.float32).view(graph.num_nodes, 1)
    graph.node_feature = torch.cat([graph.node_feature, feature], dim=-1)
################################################################################

################################################################################
# Tensor based transformation function
def tensor_pagerank(edge_index, num_nodes, alpha=0.85, max_iter=100, tol=1.0e-6):
    out_edge = {}
    for i in range(edge_index.shape[1]):
        node = edge_index[0][i].item()
        if node in out_edge:
            out_edge[node].add(edge_index[1][i].item())
        else:
            out_edge[node] = set([edge_index[1][i].item()])
    for node in range(num_nodes):
        if node not in out_edge:
            out_edge[node] = set()
    x = {}
    p = {}
    for node in range(num_nodes):
        x[node] = 1.0 / num_nodes
        p[node] = 1.0 / num_nodes
    dangling_weights = p
    dangling_nodes = [n for n in range(num_nodes) if len(out_edge[n]) == 0.0]
    for _ in range(max_iter):
        xlast = x
        x = dict.fromkeys(xlast.keys(), 0)
        danglesum = alpha * sum(xlast[n] for n in dangling_nodes)
        for n in x:
            for nbr in out_edge[n]:
                x[nbr] += alpha * xlast[n] * (1.0 / len(out_edge[n]))
            x[n] += danglesum * dangling_weights.get(n, 0) + (1.0 - alpha) * p.get(n, 0)
        err = sum([abs(x[n] - xlast[n]) for n in x])
        if err < num_nodes * tol:
            feature = torch.tensor([x[node] for node in range(num_nodes)], dtype=torch.float32).view(num_nodes, 1)
            return feature, x

def tensor_clustering_coefficient(edge_index, num_nodes):
    coefficients = {}
    edge_list = {}
    for i in range(edge_index.shape[1]):
        if edge_index[0][i].item() in edge_list:
            edge_list[edge_index[0][i].item()].add(edge_index[1][i].item())
        else:
            edge_list[edge_index[0][i].item()] = set([edge_index[1][i].item()])
    for node in range(num_nodes):
        neighbors = None
        if node in edge_list:
            neighbors = edge_list[node]
        else:
            coefficients[node] = 0
            continue
        if len(neighbors) < 2:
            coefficients[node] = 0
            continue
        count = 0
        for neighbor in neighbors:
            neighbor_neighbors = None
            if neighbor in edge_list:
                neighbor_neighbors = edge_list[neighbor]
            else:
                continue
            for neighbor_neighbor in neighbor_neighbors:
                if neighbor_neighbor in neighbors:
                    if neighbor_neighbor != node and neighbor_neighbor != neighbor:
                        count += 1
        coefficients[node] = count / len(neighbors) / (len(neighbors) - 1)
    feature = torch.tensor([coefficients[node] for node in range(num_nodes)], dtype=torch.float32).view(num_nodes, 1)
    return feature
################################################################################

def deepsnap_pagerank(args, pyg_dataset):
    avg_time = 0
    task = 'graph'
    for i in range(args.num_runs):
        if args.print_run:
            print("Run {}".format(i + 1))
        graphs = GraphDataset.pyg_to_graphs(pyg_dataset, verbose=True, fixed_split=False, netlib=netlib)
        dataset = GraphDataset(graphs, task=task)
        s = time.time()
        dataset.apply_transform(page_fun, update_tensor=False, lib=args.netlib)
        avg_time += (time.time() - s)
    print("DeepSNAP has average time: {}".format(avg_time / args.num_runs))

def deepsnap_cluster(args, pyg_dataset):
    avg_time = 0
    task = 'graph'
    for i in range(args.num_runs):
        if args.print_run:
            print("Run {}".format(i + 1))
        graphs = GraphDataset.pyg_to_graphs(pyg_dataset, verbose=True, fixed_split=False, netlib=netlib)
        dataset = GraphDataset(graphs, task=task)
        s = time.time()
        dataset.apply_transform(clustering_fun, update_tensor=False, lib=args.netlib)
        avg_time += (time.time() - s)
    print("DeepSNAP has average time: {}".format(avg_time / args.num_runs))

def pyg_pagerank(args, pyg_dataset):
    avg_time = 0
    for i in range(args.num_runs):
        if args.print_run:
            print("Run {}".format(i + 1))
        s = time.time()
        res = []
        for data in pyg_dataset:
            feature, _ = tensor_pagerank(data.edge_index, data.num_nodes)
            data.x = torch.cat([data.x, feature], dim=-1)
            res.append(data)
        avg_time += (time.time() - s)
    print("Tensor has average time: {}".format(avg_time / args.num_runs))

def pyg_cluster(args, pyg_dataset):
    avg_time = 0
    for i in range(args.num_runs):
        if args.print_run:
            print("Run {}".format(i + 1))
        s = time.time()
        res = []
        for data in pyg_dataset:
            feature = tensor_clustering_coefficient(data.edge_index, data.num_nodes)
            data.x = torch.cat([data.x, feature], dim=-1)
            res.append(data)
        avg_time += (time.time() - s)
    print("Tensor has average time: {}".format(avg_time / args.num_runs))


if __name__ == '__main__':
    args = arg_parse()

    if args.bench_task == 'page':
        print("Start benchmark PageRank!")
    else:
        print("Start benchmark Clustering coefficient!")

    if args.netlib == "nx":
        print("Use NetworkX as the DeepSNAP backend network library.")
        import networkx as netlib
    elif args.netlib == "sx":
        print("Use SnapX as the DeepSNAP backend network library.")
        import snap
        import snapx as netlib
    else:
        import networkx as netlib
        print("Use NetworkX as the DeepSNAP backend network library.")

    if args.dataset == 'COX2':
        pyg_dataset = TUDataset('./tu', args.dataset)

    if args.bench_task == 'page':
        print("Start benchmark DeepSNAP:")
        deepsnap_pagerank(args, pyg_dataset)
        print("Start benchmark Tensor:")
        pyg_pagerank(args, pyg_dataset)
    else:
        print("Start benchmark DeepSNAP:")
        deepsnap_cluster(args, pyg_dataset)
        print("Start benchmark Tensor:")
        pyg_cluster(args, pyg_dataset)
