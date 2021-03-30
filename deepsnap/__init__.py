import deepsnap.graph
import deepsnap.dataset
import deepsnap.batch
import deepsnap.hetero_graph
import deepsnap.hetero_gnn

import networkx as _netlib

__version__ = "0.2.0"

def use(netlib=None):
    r"""Specifying to use which graph library as the DeepSNAP graph backend.

    Args:
        netlib (module): The module of the graph library.

    """
    global _netlib
    if netlib is not None:
        _netlib = netlib
