import deepsnap.graph
import deepsnap.dataset
import deepsnap.batch
import deepsnap.hetero_graph
import deepsnap.hetero_gnn

import networkx as _netlib

def use(netlib=None):
    global _netlib
    if netlib is not None:
        _netlib = netlib