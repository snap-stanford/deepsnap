from collections import defaultdict

from deepsnap.batch import Batch
from deepsnap.dataset import GraphDataset
import torch
import torch.optim as optim
import torch_geometric.utils as pyg_utils
from torch_geometric.data import DataLoader
import networkx as nx

device_cache = None
def get_device():
    global device_cache
    if device_cache is None:
        device_cache = torch.device("cuda") if torch.cuda.is_available() \
            else torch.device("cpu")
    return device_cache
