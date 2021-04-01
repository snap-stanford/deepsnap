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

__version__ = "0.2.0"

def use(netlib=None):
    r"""Specifies to use which graph library as the DeepSNAP backend.

    Args:
        netlib (types.ModuleType): The module of the graph library.

    """
    global _netlib
    if netlib is not None:
        _netlib = netlib

def set_seed(seed):
	r"""
	Sets seeds to generate random numbers. This function will set seeds for :obj:`random`, 
	:obj:`numpy.random`, :obj:`torch.manual_seed`, and :obj:`torch.cuda.manual_seed_all`.

    Args:
        seed (int): The seed value to generate random numbers.

    """
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)