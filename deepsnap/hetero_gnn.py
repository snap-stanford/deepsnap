import torch
from torch._six import container_abcs
from torch_geometric.nn.inits import reset
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
import torch.nn as nn


class HeteroSAGEConv(pyg_nn.MessagePassing):
    r"""The heterogeneous compitable GraphSAGE operator is derived from the `"Inductive Representation
    Learning on Large Graphs" <https://arxiv.org/abs/1706.02216>`_, `"Modeling polypharmacy side
    effects with graph convolutional networks" <https://arxiv.org/abs/1802.00543>`_ and `"Modeling
    Relational Data with Graph Convolutional Networks" <https://arxiv.org/abs/1703.06103>`_ papers.

    Args:
        in_channels_neigh (int): The input dimension of the end node type.
        out_channels (int): The dimension of the output.
        in_channels_self (int): The input dimension of the start node type.
            Default is `None` where the `in_channels_self` is equal to `in_channels_neigh`.
    """
    def __init__(self, in_channels_neigh, out_channels, in_channels_self=None):
        super(HeteroSAGEConv, self).__init__(aggr="add")
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
        """"""
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        return self.propagate(
            edge_index, size=size,
            node_feature_neigh=node_feature_neigh,
            node_feature_self=node_feature_self,
            edge_weight=edge_weight, res_n_id=res_n_id
        )

    def message(self, node_feature_neigh_j, node_feature_self_i, edge_weight):
        """"""
        return node_feature_neigh_j

    def update(self, aggr_out, node_feature_self, res_n_id):
        """"""
        aggr_out = self.lin_neigh(aggr_out)
        node_feature_self = self.lin_self(node_feature_self)
        aggr_out = torch.cat([aggr_out, node_feature_self], dim=-1)
        aggr_out = self.lin_update(aggr_out)
        return aggr_out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            "(neigh: {self.in_channels_neigh}, self: {self.in_channels_self}, "
            "out: {self.out_channels})"
        )


class HeteroConv(torch.nn.Module):
    r"""A "wrapper" layer designed for heterogeneous graph layers. It takes a
    heterogeneous graph layer, such as :class:`deepsnap.hetero_gnn.HeteroSAGEConv`, at the initializing stage.
    """
    def __init__(self, convs, aggr="add", parallelize=False):
        super(HeteroConv, self).__init__()

        assert isinstance(convs, container_abcs.Mapping)
        self.convs = convs
        self.modules = torch.nn.ModuleList(convs.values())

        assert aggr in ["add", "mean", "max", "mul", "concat", None]
        self.aggr = aggr

        if parallelize and torch.cuda.is_available():
            self.streams = {key: torch.cuda.Stream() for key in convs.keys()}
        else:
            self.streams = None

    def reset_parameters(self):
        for conv in self.convs.values():
            reset(conv)

    def forward(self, node_features, edge_indices, edge_features=None):
        r"""The forward function for `HeteroConv`.

        Args:
            node_features (dict): A dictionary each key is node type and the corresponding
                value is a node feature tensor.
            edge_indices (dict): A dictionary each key is message type and the corresponding
                value is an edge index tensor.
            edge_features (dict): A dictionary each key is edge type and the corresponding
                value is an edge feature tensor. Default is `None`.
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

    def aggregate(self, xs):
        r"""The aggregation for each node type. Currently support `concat`, `add`,
        `mean`, `max` and `mul`.
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


def forward_op(x, func, **kwargs):
    r"""A helper function for the heterogeneous operations. Given a dictionary input,
    it will return a dictionary with the same keys and the values applied by the
    `func` with specified parameters.

    Args:
        x (dict): A dictionary that the value of each item will be applied by the `func`.
        func (:class:`function`): The function will be applied to each value in the dictionary.
        **kwargs: Parameters that will be passed into the `func`.
    """
    if not isinstance(x, dict):
        raise ValueError("The input x should be a dictionary")
    for key in x:
        x[key] = func(x[key], **kwargs)
    return x


def loss_op(pred, y, label_index, loss_func, **kwargs):
    r"""A helper function for the heterogeneous loss operations.

    Args:
        pred (dict): A dictionary of predictions.
        y (dict): A dictionary of labels.
        label_index (dict): A dictionary of indicies that the loss will be computed on.
            Each value should be a Pytorch long tensor.
        loss_func (:class:`function`): The loss function.
        **kwargs: Parameters that will be passed into the `loss_func`.
    """
    loss = 0
    for key in pred:
        idx = label_index[key]
        loss += loss_func(pred[key][idx], y[key][idx])
    return loss
