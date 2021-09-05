import types
import torch
import random
import numpy as np
import deepsnap.graph
import deepsnap.dataset
import deepsnap.batch
import deepsnap.hetero_graph
import deepsnap.hetero_gnn

import networkx as _netlib

__version__ = "0.2.1"

def use(netlib=None):
    r"""
    Specifies to use which graph library as the DeepSNAP backend.
    The default backend is NetworkX. Current DeepSNAP also supports 
    Snap.py (SnapX) backend for undirected homogeneous graph. You 
    can switch the backend to SnapX via:

    .. code-block:: python

        import snap
        import snapx as sx
        import deepsnap
        deepsnap.use(sx)

    Args:
        netlib (types.ModuleType, optional): The graph backend module. 
            Currently DeepSNAP supports the NetworkX and SnapX (for 
            SnapX only the undirected homogeneous graph) as the graph 
            backend. Default graph backend is the NetworkX.

    """
    global _netlib
    if netlib is not None:
        _netlib = netlib

def set_seed(seed):
    r"""
    Sets seeds to generate random numbers. This function will set seeds 
    of :obj:`random.seed`, :obj:`numpy.random.seed`, 
    :obj:`torch.manual_seed`, and :obj:`torch.cuda.manual_seed_all` to 
    be the `seed`.

    Use the function in following way:

    .. code-block:: python

        import deepsnap
        deepsnap.set_seed(1)

    Args:
        seed (int): The seed value to generate random numbers.

    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
