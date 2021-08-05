import collections.abc
import torch
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
import torch.nn as nn

from torch import Tensor
from torch_geometric.nn.inits import reset
from torch_sparse import matmul
from typing import (
    List,
    Dict,
)

# TODO: add another new "HeteroSAGEConv" add edge_features
class HeteroSAGEConv(pyg_nn.MessagePassing):
    r"""The heterogeneous compitable GraphSAGE operator is derived from the `"Inductive Representation
    Learning on Large Graphs" <https://arxiv.org/abs/1706.02216>`_, `"Modeling polypharmacy side
    effects with graph convolutional networks" <https://arxiv.org/abs/1802.00543>`_, and `"Modeling
    Relational Data with Graph Convolutional Networks" <https://arxiv.org/abs/1703.06103>`_ papers.

    .. note::

        This layer is usually used with the :class:`HeteroConv`.

    Args:
        in_channels_neigh (int): The input dimension of the neighbor node type.
        out_channels (int): The dimension of the output.
        in_channels_self (int): The input dimension of the self node type.
            Default is `None` where the `in_channels_self` is equal to `in_channels_neigh`.
        remove_self_loop (bool): Whether to remove self loops using :class:`torch_geometric.utils.remove_self_loops`.
            Default is `True`.
    """
    def __init__(self, in_channels_neigh, out_channels, in_channels_self=None, remove_self_loop=True):
        super(HeteroSAGEConv, self).__init__(aggr="add")
        self.remove_self_loop = remove_self_loop
        self.in_channels_neigh = in_channels_neigh
        if in_channels_self is None:
            self.in_channels_self = in_channels_neigh
        else:
            self.in_channels_self = in_channels_self
        self.out_channels = out_channels
        self.lin_neigh = nn.Linear(self.in_channels_neigh, self.out_channels)
        self.lin_self = nn.Linear(self.in_channels_self, self.out_channels)
        self.lin_update = nn.Linear(self.out_channels * 2, self.out_channels)

    def forward(
        self,
        node_feature_neigh,
        node_feature_self,
        edge_index,
        edge_weight=None,
        size=None,
        res_n_id=None,
    ):
        r"""
        """
        if self.remove_self_loop:
            edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        return self.propagate(
            edge_index, size=size,
            node_feature_neigh=node_feature_neigh,
            node_feature_self=node_feature_self,
            edge_weight=edge_weight, res_n_id=res_n_id
        )

    def message(self, node_feature_neigh_j, node_feature_self_i, edge_weight):
        r"""
        """
        return node_feature_neigh_j
        # torch.cat([node_feature_self_j, edge_feature, node_feature_self_i], dim=...)
        # TODO: check out homogenous wordnet message passing

    def message_and_aggregate(self, edge_index, node_feature_neigh):
        r"""
        This function basically fuses the :meth:`message` and :meth:`aggregate` into 
        one function. It will save memory and avoid message materialization. More 
        information please refer to the PyTorch Geometric documentation.

        Args:
            edge_index (:class:`torch_sparse.SparseTensor`): The `edge_index` sparse tensor.
            node_feature_neigh (:class:`torch.Tensor`): Neighbor feature tensor.
        """
        out = matmul(edge_index, node_feature_neigh, reduce="mean")
        return out

    def update(self, aggr_out, node_feature_self, res_n_id):
        r"""
        """
        aggr_out = self.lin_neigh(aggr_out)
        node_feature_self = self.lin_self(node_feature_self)
        aggr_out = torch.cat([aggr_out, node_feature_self], dim=-1)
        aggr_out = self.lin_update(aggr_out)
        return aggr_out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            f"(neigh: {self.in_channels_neigh}, self: {self.in_channels_self}, "
            f"out: {self.out_channels})"
        )


class HeteroConv(torch.nn.Module):
    r"""A "wrapper" layer designed for heterogeneous graph layers. It takes a
    heterogeneous graph layer, such as :class:`deepsnap.hetero_gnn.HeteroSAGEConv`, 
    at the initializing stage. Currently DeepSNAP does not support `parallelize=True`.

    .. note::

        For more detailed use of :class:`HeteroConv`, see the `examples/node_classification_hetero 
        <https://github.com/snap-stanford/deepsnap/tree/master/examples/node_classification_hetero>`_ 
        folder.

    """
    def __init__(self, convs, aggr="add", parallelize=False):
        super(HeteroConv, self).__init__()

        assert isinstance(convs, collections.abc.Mapping)
        self.convs = convs
        self.modules = torch.nn.ModuleList(convs.values())

        assert aggr in ["add", "mean", "max", "mul", "concat", None]
        self.aggr = aggr

        if parallelize and torch.cuda.is_available():
            self.streams = {key: torch.cuda.Stream() for key in convs.keys()}
        else:
            self.streams = None

    def reset_parameters(self):
        r"""
        """
        for conv in self.convs.values():
            reset(conv)

    def forward(self, node_features, edge_indices, edge_features=None):
        r"""The forward function for :class:`HeteroConv`.

        Args:
            node_features (Dict[str, Tensor]): A dictionary each key is node type and the corresponding
                value is a node feature tensor.
            edge_indices (Dict[str, Tensor]): A dictionary each key is message type and the corresponding
                value is an `edge _ndex` tensor.
            edge_features (Dict[str, Tensor]): A dictionary each key is edge type and the corresponding
                value is an edge feature tensor. The default value is `None`.
        """
        # TODO: graph is not defined
        if self.streams is not None and graph.not_in_gpu():
            raise RuntimeError("Cannot parallelize on non-gpu graphs")

        # node embedding computed from each message type
        message_type_emb = {}
        for message_key, message_type in edge_indices.items():
            if message_key not in self.convs:
                continue
            neigh_type, edge_type, self_type = message_key
            node_feature_neigh = node_features[neigh_type]
            node_feature_self = node_features[self_type]
            # TODO: edge_features is not used
            if edge_features is not None:
                edge_feature = edge_features[edge_type]
            edge_index = edge_indices[message_key]

            # Perform message passing.
            if self.streams is not None:
                with torch.cuda.stream(self.streams[message_key]):
                    message_type_emb[message_key] = (
                        self.convs[message_key](
                            node_feature_neigh,
                            node_feature_self,
                            edge_index,
                        )
                    )
            else:
                message_type_emb[message_key] = (
                    self.convs[message_key](
                        node_feature_neigh,
                        node_feature_self,
                        edge_index,
                    )
                )

        if self.streams is not None:
            torch.cuda.synchronize()

        # aggregate node embeddings from different message types into 1 node
        # embedding for each node
        node_emb = {tail: [] for _, _, tail in message_type_emb.keys()}

        for (_, _, tail), item in message_type_emb.items():
            node_emb[tail].append(item)

        # Aggregate multiple embeddings with the same tail.
        for node_type, embs in node_emb.items():
            if len(embs) == 1:
                node_emb[node_type] = embs[0]
            else:
                node_emb[node_type] = self.aggregate(embs)

        return node_emb

    def aggregate(self, xs: List[Tensor]):
        r"""The aggregation for each node type. Currently support `concat`, `add`,
        `mean`, `max` and `mul`.

        Args:
            xs (List[Tensor]): A list of :class:`torch.Tensor` for a node type. 
                The number of elements in the list equals to the number of 
                `message types` that the destination node type is current node type.
        """
        if self.aggr == "concat":
            return torch.cat(xs, dim=-1)

        x = torch.stack(xs, dim=-1)
        if self.aggr == "add":
            return x.sum(dim=-1)
        elif self.aggr == "mean":
            return x.mean(dim=-1)
        elif self.aggr == "max":
            return x.max(dim=-1)[0]
        elif self.aggr == "mul":
            return x.prod(dim=-1)[0]


def forward_op(x, module_dict, **kwargs):
    r"""A helper function for the heterogeneous operations. Given a dictionary input
    `x`, it will return a dictionary with the same keys and the values applied by the
    corresponding values of the `module_dict` with specified parameters. The keys in `x` 
    are same as the keys in the `module_dict`. If `module_dict` is not a dictionary,
    it is assumed to be a single value.

    Args:
        x (Dict[str, Tensor]): A dictionary that the value of each item is a tensor.
        module_dict (:class:`torch.nn.ModuleDict`): The value of the `module_dict` 
            will be fed with each value in `x`.
        **kwargs (optional): Parameters that will be passed into each value of the 
            `module_dict`.
    """
    if not isinstance(x, dict):
        raise ValueError("The input x should be a dictionary.")
    res = {}
    if not isinstance(module_dict, dict) and not isinstance(module_dict, nn.ModuleDict):
        for key in x:
            res[key] = module_dict(x[key], **kwargs)
    else:
        for key in x:
            res[key] = module_dict[key](x[key], **kwargs)
    return res


def loss_op(pred, y, index, loss_func):
    r"""
    A helper function for the heterogeneous loss operations.
    This function will sum the loss of all node types.

    Args:
        pred (Dict[str, Tensor]): A dictionary of prediction results.
        y (Dict[str, Tensor]): A dictionary of labels. The keys should match with 
            the keys in the `pred`.
        index (Dict[str, Tensor]): A dictionary of indicies that the loss 
            will be computed on. Each value should be :class:`torch.LongTensor`. 
            Notice that `y` will not be indexed by the `index`. Here we assume 
            `y` has been splitted into proper sets.
        loss_func (callable): The defined loss function.
    """
    loss = 0
    for node_type in pred:
        idx = index[node_type]
        loss += loss_func(pred[node_type][idx], y[node_type])
    return loss
