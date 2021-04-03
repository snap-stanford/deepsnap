import re
import types
import random
import copy
import math
import pdb
import numpy as np
import torch
from torch_geometric.utils import to_undirected
from typing import (
    Dict,
    List,
    Union
)
import warnings
import deepsnap


class Graph(object):
    r"""
    A plain python object modeling a single graph with various
    (optional) attributes.

    Args:
        G (Graph object, optional): The NetworkX or SnapX graph 
            object which contains features and labels. If it is not 
            specified, :class:`Graph` will use the tensor backend. 
        netlib (types.ModuleType, optional): The graph backend module. 
                Currently DeepSNAP supports the NetworkX and SnapX (for 
                SnapX only the undirected homogeneous graph) as the graph 
                backend. Default graph backend is the NetworkX.
        **kwargs (optional): Keyworded argument list with keys such
            as :obj:`node_feature`, :obj:`node_label` and 
            values that are corresponding attributes. The features 
            are required for the tensor backend.
    """

    def __init__(self, G=None, netlib=None, **kwargs):
        self.G = G
        if netlib is not None:
            deepsnap._netlib = netlib
        keys = [
            "node_feature",
            "node_label",
            "edge_feature",
            "edge_label",
            "graph_feature",
            "graph_label",
            "edge_index",
            "edge_label_index",
            "node_label_index",
            "custom"
        ]
        for key in keys:
            self[key] = None

        self._is_train = False
        self._num_positive_examples = None

        for key, item in kwargs.items():
            self[key] = item

        if G is None and kwargs:
            if "directed" not in kwargs:
                self.directed = True
            if "edge_index" not in kwargs:
                raise ValueError(
                    "A tensor of edge_index is required by using "
                    "the tensor backend."
                )
            # check for undirected edge_index format
            if not self.directed:
                edge_index_length = self.edge_index.shape[1]
                edge_index_first_half, _ = (
                    torch.sort(self.edge_index[:, :int(edge_index_length / 2)])
                )
                edge_index_second_half, _ = (
                    torch.sort(self.edge_index[:, int(edge_index_length / 2):])
                )
                if not torch.equal(
                    edge_index_first_half,
                    torch.flip(edge_index_second_half, [0])
                ):
                    raise ValueError(
                        "In tensor backend mode with undirected graph, "
                        "the user provided edge_index should contain "
                        "undirected edges for both directions."
                        "the first half of edge_index should contain "
                        "unique edges in one direction and the second "
                        "half of edge_index should contain the same set "
                        "of unique edges of another direction."
                    )

        if G is not None or kwargs:
            # handle tensor backend + custom support
            if (
                ("edge_label_index" not in kwargs)
                and ("node_label_index" not in kwargs)
            ):
                self._update_tensors(init=True)
            else:
                self._update_tensors(init=False)

    @classmethod
    def _from_dict(cls, dictionary: Dict[str, torch.tensor]):
        r"""
        Creates a data object from a python dictionary.

        Args:
            dictionary (dict): Python dictionary with key (string)
            - value (torch.tensor) pair.

        Returns:
            :class:`deepsnap.graph.Graph`: return a new Graph object
            with the data from the dictionary.
        """
        if "G" in dictionary:
            # If there is an G, initialize class in the graph backend
            graph = cls(G=dictionary["G"])
        else:
            graph = cls(**dictionary)
        for key, item in dictionary.items():
            graph[key] = item

        return graph

    def __getitem__(self, key: str):
        r"""
        Gets the data of the attribute :obj:`key`.
        """
        return getattr(self, key, None)

    def __setitem__(self, key: str, value):
        """Sets the attribute :obj:`key` to :obj:`value`."""
        setattr(self, key, value)

    @property
    def keys(self):
        r"""
        Returns all names of the graph attributes.

        Returns:
            list: List of attributes in the :class:`Graph` object.
        """
        # filter attributes that are not observed by users
        # (1) those with value "None"; (2) those start with '_'
        keys = [
            key
            for key in self.__dict__.keys()
            if self[key] is not None and key[0] != "_"
        ]
        return keys

    def __len__(self) -> int:
        r"""
        Returns the number of all present attributes.

        Returns:
            int: The number of all present attributes.
        """
        return len(self.keys)

    def __contains__(self, key: str) -> bool:
        r"""
        Returns :obj:`True`, if the attribute :obj:`key` is present in the
        data.
        """
        return key in self.keys

    def __iter__(self):
        r"""
        Iterates over all present attributes in the data, yielding their
        attribute names and content.
        """
        for key in sorted(self.keys):
            yield key, self[key]

    def __call__(self, *keys):
        r"""
        Iterates over all attributes :obj:`*keys` in the data, yielding
        their attribute names and content.
        If :obj:`*keys` is not given this method will iterative over all
        present attributes.
        """
        for key in sorted(self.keys) if not keys else keys:
            if key in self:
                yield key, self[key]

    def __cat_dim__(self, key: str, value) -> int:
        r"""
        Returns the dimension for which :obj:`value` of attribute
        :obj:`key` will get concatenated when creating batches.

        .. note::

            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        # `*index*` and `*face*` should be concatenated in the last dimension,
        # everything else in the first dimension.
        return -1 if "index" in key else 0

    def __inc__(self, key: str, value) -> int:
        r""""
        Returns the incremental count to cumulatively increase the value
        of the next attribute of :obj:`key` when creating batches.

        .. note::

            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        # Only `*index*` and `*face*` should be cumulatively summed up when
        # creating batches.
        return self.num_nodes if "index" in key else 0

    @property
    def num_nodes(self) -> int:
        r"""
        Return number of nodes in the graph.

        Returns:
            int: Number of nodes in the graph.
        """
        if self.G is not None:
            return self.G.number_of_nodes()
        return self[self._node_related_key].shape[0]

    @property
    def num_edges(self) -> int:
        r"""
        Returns the number of edges in the graph.

        Returns:
            int: Number of edges in the graph.
        """
        if self.G is not None:
            return self.G.number_of_edges()
        if self.is_undirected():
            return int(self.edge_index.shape[1] / 2)
        return self.edge_index.shape[1]

    @property
    def num_node_features(self) -> int:
        r"""
        Returns node feature dimension in the graph.

        Returns:
            int: Node feature dimension. `0` if there is no 
            `node_feature`.
        """
        return self.get_num_dims("node_feature", as_label=False)

    @property
    def num_node_labels(self) -> int:
        r"""
        Returns the number of the node labels in the graph.

        Returns:
            int: Number of node labels. `0` if there is no 
            `node_label`.
        """
        return self.get_num_dims("node_label", as_label=True)

    @property
    def num_edge_features(self) -> int:
        r"""
        Returns edge feature dimension in the graph.

        Returns:
            int: Edge feature dimension. `0` if there is no 
            `edge_feature`.
        """
        return self.get_num_dims("edge_feature", as_label=False)

    @property
    def num_edge_labels(self) -> int:
        r"""
        Returns the number of the edge labels in the graph.

        Returns:
            int: Number of edge labels. `0` if there is no 
            `edge_label`.
        """
        return self.get_num_dims("edge_label", as_label=True)

    @property
    def num_graph_features(self) -> int:
        r"""
        Returns graph feature dimension in the graph.

        Returns:
            int: Graph feature dimension. `0` if there is no
            `graph_feature`.
        """
        return self.get_num_dims("graph_feature", as_label=False)

    @property
    def num_graph_labels(self) -> int:
        r"""
        Returns the number of the graph labels in the graph.

        Returns:
            int: Number of graph labels. `0` if there is no 
            `graph_label`.
        """
        return self.get_num_dims("graph_label", as_label=True)

    def get_num_labels(self, key: str):
        r"""
        Gets the lables for a specified key.

        Args:
            key (str): The chosen property.

        Returns:
            :class:`torch.Tensor`: Unique lables (in tensor format).
        """
        return torch.unique(self[key])

    def get_num_dims(self, key: str, as_label: bool = False) -> int:
        r"""
        Returns the number of dimensions for one graph/node/edge property.

        Args:
            key (str): The chosen property.
            as_label (bool): If `as_label`, treat the tensor as labels.

        Returns:
            int: The number of dimensions for chosen property.
        """
        if as_label:
            # treat as label
            if self[key] is not None:
                if self[key].dtype == torch.long:
                    # classification label
                    return self.get_num_labels(key).shape[0]
                else:
                    # regression label
                    if (
                        len(self[key].shape) == 1
                        and not Graph._is_graph_attribute(key)
                    ):
                        # for node/edge tasks: 1 scalar per node/edge
                        return 1
                    else:
                        return self[key].shape[-1]
            else:
                return 0
        else:
            # treat as feature
            if self[key] is not None:
                return self[key].shape[1]
            else:
                return 0

    def is_directed(self) -> bool:
        r"""
        Whether the graph is directed.

        Returns:
            bool: `True` if the graph is directed.
        """
        if self.G is not None:
            return self.G.is_directed()
        return self.directed

    def is_undirected(self) -> bool:
        r"""
        Whether the graph is undirected.

        Returns:
            bool: `True` if the graph is undirected.
        """
        return not self.is_directed()

    def apply_tensor(self, func, *keys):
        r"""
        Applies the function :obj:`func` to all tensor attributes specified by
        :obj:`*keys`. If the :obj:`*keys` is not given, :obj:`func` is applied to
        all present attributes.

        Args:
            func (callable): The function that will be applied 
                to a PyTorch tensor(s).
            *keys (str, optional): Names of the tensor attributes that will
                be applied.

        Returns:
            :class:`deepsnap.graph.Graph`: Return the
            self :class:`deepsnap.graph.Graph`.
        """
        for key, item in self(*keys):
            if torch.is_tensor(item):
                self[key] = func(item)
            elif isinstance(self[key], dict):
                for obj_key, obj_item in self[key].items():
                    if torch.is_tensor(obj_item):
                        self[key][obj_key] = func(obj_item)
        return self

    def contiguous(self, *keys):
        r"""
        Ensures a contiguous memory layout for the attributes specified by
        :obj:`*keys`. If :obj:`*keys` is not given, all present attributes
        are ensured to have a contiguous memory layout.

        Args:
            *keys (str, optional): Tensor attributes which will be in
                contiguous memory layout.

        Returns:
            :class:`Graph`: :class:`Graph` object with specified tensor 
            attributes in contiguous memory layout.
        """
        return self.apply_tensor(lambda x: x.contiguous(), *keys)

    def to(self, device, *keys):
        r"""
        Transfers tensor to specified device for to all attributes that 
        are specified in the :obj:`*keys`.
        If :obj:`*keys` is not given, the conversion is applied to all 
        present attributes.

        Args:
            device (str): Specified device name, such as `cpu` or
                `cuda`.
            *keys (str, optional): Tensor attributes that will be 
                transferred to specified device.
        """
        return self.apply_tensor(lambda x: x.to(device), *keys)

    def clone(self):
        r"""
        Deepcopy the graph object.

        Returns:
            :class:`Graph`:
            A cloned :class:`Graph` object with deepcopying
            all features.
        """
        dictionary = {}
        for k, v in self.__dict__.items():
            if torch.is_tensor(v):
                dictionary[k] = v.clone()
            elif k == "netlib":
                dictionary[k] = v
            else:
                if hasattr(v, "netlib"):
                    v.netlib = None
                dictionary[k] = copy.deepcopy(v)
        return self.__class__._from_dict(dictionary)

    def _size_repr(self, value) -> List[int]:
        r"""
        Returns:
            list: A list of size of each element in value
        """
        if torch.is_tensor(value):
            return list(value.size())
        elif isinstance(value, int) or isinstance(value, float):
            return [1]
        elif isinstance(value, list) or isinstance(value, tuple):
            return [len(value)]
        else:
            return []

    def __repr__(self):
        info = [f"{key}={self._size_repr(item)}" for key, item in self]
        return f"{self.__class__.__name__}({', '.join(info)})"

    @staticmethod
    def _is_edge_attribute(key: str) -> bool:
        r"""
        Check whether an attribute is a edge attribute.
        """
        # could be feature, label, etc.
        return "edge" in key and "index" not in key

    @staticmethod
    def _is_node_attribute(key: str) -> bool:
        r"""
        Check whether an attribute is a node attribute.
        """
        # could be feature, label, etc.
        return "node" in key and "index" not in key

    @staticmethod
    def _is_graph_attribute(key: str) -> bool:
        r"""
        Check whether an attribute is a graph attribute.
        """
        return "graph" in key and "index" not in key

    def _is_valid(self):
        r"""
        Check validity.
        """
        for key in self.keys:
            if self._is_node_attribute(key):
                if self.num_nodes != self[key].shape[0]:
                    raise ValueError(
                        f"key {key} is not valid, num nodes must equal "
                        "num nodes w/ features."
                    )

    def _update_tensors(self, init: bool = False):
        r"""
        Update attributes and indices with values from the self.G
        graph object.
        """
        if self.G is not None:
            self._update_attributes()
        self._node_related_key = None
        for key in self.keys:
            if self._is_node_attribute(key):
                self._node_related_key = key
                break
        if self._node_related_key is None:
            warnings.warn("Node related key is required.")

        self._update_index(init)

    def _update_attributes(self):
        r"""
        Update attributes with values from the self.g graph object.
        """
        # node
        if self.G.number_of_nodes() == 0:
            raise ValueError(
                "in _update_attributes, number of nodes in Graph "
                "G must be larger than 0"
            )
        if self.G.number_of_edges() == 0:
            raise ValueError(
                "in _update_attributes, number of edges in Graph "
                "G must be larger than 0"
            )

        # node
        keys = next(iter(self.G.nodes(data=True)))[-1].keys()
        for key in keys:
            self[key] = self._get_node_attributes(key)
        # edge
        keys = next(iter(self.G.edges(data=True)))[-1].keys()
        for key in keys:
            self[key] = self._get_edge_attributes(key)
        # graph
        keys = self.G.graph.keys()
        for key in keys:
            self[key] = self._get_graph_attributes(key)

    def _get_node_attributes(self, name: str) -> torch.tensor:
        r"""
        Returns the node attributes in the graph.
        Multiple attributes will be stacked.

        Args:
            name(string): the name of the attributes to return.

        Returns:
            :class:`torch.tensor`: Node attributes.
        """
        # new: concat
        attributes = []
        for _, d in self.G.nodes.items():
            if name in d:
                attributes.append(d[name])
        if len(attributes) == 0:
            return None
        if torch.is_tensor(attributes[0]):
            attributes = torch.stack(attributes, dim=0)
        elif isinstance(attributes[0], float):
            attributes = torch.tensor(attributes, dtype=torch.float)
        elif isinstance(attributes[0], int):
            attributes = torch.tensor(attributes, dtype=torch.long)

        return attributes

    def _get_edge_attributes(self, key: str) -> torch.tensor:
        r"""
        Returns the edge attributes in the graph.
        Multiple attributes will be stacked.

        Args:
            key(string): the name of the attributes to return.

        Returns:
            :class:`torch.tensor`: Edge attributes.
        """

        # new: concat
        attributes = []
        for x in self.G.edges(data=True):
            if key in x[-1]:
                attributes.append(x[-1][key])
        if len(attributes) == 0:
            return None
        if torch.is_tensor(attributes[0]):
            attributes = torch.stack(attributes, dim=0)
        elif isinstance(attributes[0], float):
            attributes = torch.tensor(attributes, dtype=torch.float)
        elif isinstance(attributes[0], int):
            attributes = torch.tensor(attributes, dtype=torch.long)
        else:
            raise TypeError(f"Unknown type {key} in edge attributes.")

        if self.is_undirected():
            attributes = torch.cat([attributes, attributes], dim=0)

        return attributes

    def _get_graph_attributes(self, key: str):
        r"""
        Returns the graph attributes.

        Args:
            key(string): the name of the attributes to return.

        Returns:
            any: graph attributes with the specified name.
        """
        return self.G.graph.get(key)

    def _update_nodes(
        self,
        nodes,
        mapping: Dict[Union[str, int], int]
    ) -> List[tuple]:
        r"""
        Relabel nodes following mapping and add node dictionary for each
        node if it is not already provided.

        Returns:
            list: A list of tuples representing nodes and node dictionaries.
        """

        if isinstance(nodes[0], tuple):
            # node dictionary is already provided
            nodes = [
                (mapping[node[0]], node[-1])
                for node in nodes
            ]
        else:
            # node dictionary is not provided
            nodes = [
                (
                    mapping[node],
                    self.G.nodes[mapping[node]]
                )
                for node in nodes
            ]
        return nodes

    def _update_edges(self, edges, mapping, add_edge_info: bool = True):
        r"""
        Relabel edges following mapping and add edge dictionary for each
        edge if it is not already provided.

        Returns:
            list: A list of tuples representing edges and edge dictionaries.
        """
        for i in range(len(edges)):
            node_0 = mapping[
                edges[i][0]
            ]
            node_1 = mapping[
                edges[i][1]
            ]

            if isinstance(edges[i][-1], dict):
                # edge dictionary is already provided
                edge_info = edges[i][-1]
                if len(edges[i][:-1]) == 2:
                    edge = (node_0, node_1, edge_info)
                elif len(edges[i][:-1]) == 3:
                    graph_index = edges[i][2]
                    edge = (node_0, node_1, graph_index, edge_info)
                else:
                    raise ValueError("Each edge has more than 3 indices.")
            else:
                # edge dictionary is not provided
                if len(edges[i]) == 2:
                    # not multigraph
                    if add_edge_info:
                        if self.G is not None:
                            edge = (
                                node_0, node_1, self.G.edges[node_0, node_1]
                            )
                        else:
                            feature_dict = {}
                            for key in self.keys:
                                if (
                                    self._is_edge_attribute(key)
                                    and torch.is_tensor(self[key])
                                ):
                                    feature_dict[key] = self[key][i]
                            edge = (node_0, node_1, feature_dict)
                    else:
                        edge = (node_0, node_1)
                elif len(edges[i]) == 3:
                    # multigraph
                    graph_index = edges[i][2]
                    if add_edge_info:
                        if self.G is not None:
                            edge = (
                                node_0, node_1, graph_index,
                                self.G.edges[node_0, node_1, graph_index]
                            )
                        else:
                            feature_dict = {}
                            for key in self.keys:
                                if (
                                    self._is_edge_attribute(key)
                                    and torch.is_tensor(self[key])
                                ):
                                    feature_dict[key] = self[key][i]
                            edge = (node_0, node_1, feature_dict)
                    else:
                        edge = (node_0, node_1, graph_index)
                else:
                    raise ValueError("Each edge has more than 3 indices.")

            edges[i] = edge
        return edges

    def _custom_update(self, mapping: Dict[Union[int, str], int]):
        r"""
        Custom support by populating self.general_splits,
        self.disjoint_split self.negative_edges and self.task
        """
        custom_keys = [
            "general_splits", "disjoint_split", "negative_edges", "task"
        ]
        if self.custom is not None:
            for custom_key in custom_keys:
                if custom_key in self.custom:
                    self[custom_key] = self.custom[custom_key]
                elif not hasattr(self, custom_key):
                    self[custom_key] = None

            if self.task is None:
                raise ValueError(
                    "User must provide the task variable in dataset or graph "
                    "custom. optional values for task are node, edge and "
                    "link_pred."
                )
            if self.task not in ["node", "edge", "link_pred"]:
                raise ValueError(
                    "self.task in graph.py must be either node, "
                    "edge or link_pred. the current self.task "
                    f"value is {self.task}."
                )
            if self.general_splits is not None:
                if self.task == "node":
                    for i in range(len(self.general_splits)):
                        self.general_splits[i] = self._update_nodes(
                            self.general_splits[i],
                            mapping
                        )
                elif self.task == "edge" or self.task == "link_pred":
                    for i in range(len(self.general_splits)):
                        self.general_splits[i] = self._update_edges(
                            self.general_splits[i],
                            mapping
                        )

            if self.disjoint_split is not None:
                if self.task == "link_pred":
                    self.disjoint_split = self._update_edges(
                        self.disjoint_split,
                        mapping
                    )
                else:
                    raise ValueError(
                        "When self.disjoint_splits is not "
                        "None, self.task must be `link_pred`."
                    )

            if self.negative_edges is not None:
                if self.task == "link_pred":
                    for i in range(len(self.negative_edges)):
                        self.negative_edges[i] = self._update_edges(
                            self.negative_edges[i],
                            mapping,
                            add_edge_info=False
                        )
                else:
                    raise ValueError(
                        "When self.negative_edges is not "
                        "None, self.task must be `link_pred`."
                    )

            self._custom_update_flag = True

    def _update_index(self, init: bool = False):
        r"""
        Update attributes and indices with values from the self.G
        """
        # relabel graphs
        if self.G is not None:
            keys = list(self.G.nodes)
            vals = list(range(self.num_nodes))
            mapping = dict(zip(keys, vals))
            if keys != vals:
                self.G = deepsnap._netlib.relabel_nodes(
                    self.G, mapping, copy=True
                )
            # get edges
            self.edge_index = self._edge_to_index(list(self.G.edges))
        else:
            mapping = {x: x for x in range(self.num_nodes)}
        if init:
            # init is only true when creating the variables
            # edge_label_index and node_label_index
            self.edge_label_index = copy.deepcopy(self.edge_index)
            self.node_label_index = (
                torch.arange(self.num_nodes, dtype=torch.long)
            )

            self._custom_update(mapping)

    def _node_to_index(self, nodes):
        r"""
        List of G.nodes to torch tensor node_index

        Only the selected nodes' node indices are extracted.

        Returns:
            :class:`torch.tensor`: Node indices.
        """
        nodes = [node[0] for node in nodes]
        node_index = torch.tensor(nodes)
        return node_index

    def _edge_to_index(self, edges):
        r"""
        List of G.edges to torch tensor edge_index

        Only the selected edges' edge indices are extracted.

        Returns:
            :class:`torch.tensor`: Edge indices.
        """
        edges = [(edge[0], edge[1]) for edge in edges]

        edge_index = torch.tensor(edges)
        if self.is_undirected():
            edge_index = torch.cat(
                [edge_index, torch.flip(edge_index, [1])],
                dim=0
            )
        return edge_index.permute(1, 0)

    def _get_edge_attributes_by_key(self, edges, key: str):
        r"""
        List of G.edges to torch tensor for key,
        with dimension [num_edges x key_dim].

        Only the selected edges' attributes are extracted.

        Returns:
            :class:`torch.tensor`: Edge attributes.
        """
        if len(edges) == 0:
            raise ValueError(
                "in _get_edge_attributes_by_key, "
                "len(edges) must be larger than 0."
            )
        if not isinstance(edges[0][-1], dict) or key not in edges[0][-1]:
            return None

        attributes = [edge[-1][key] for edge in edges]
        if torch.is_tensor(attributes[0]):
            attributes = torch.stack(attributes, dim=0)
        elif isinstance(attributes[0], float):
            attributes = torch.tensor(attributes, dtype=torch.float)
        elif isinstance(attributes[0], int):
            attributes = torch.tensor(attributes, dtype=torch.long)
        else:
            raise ValueError("Unknown type of key {} in edge attributes.")

        if self.is_undirected():
            attributes = torch.cat([attributes, attributes], dim=0)

        return attributes

    def _get_edge_attributes_by_key_tensor(self, edge_index, key: str):
        r"""
        Extract the edge attributes indicated by edge_index in tensor backend.

        Returns:
            :class:`torch.tensor`: Edge attributes.
        """
        if not torch.is_tensor(edge_index):
            raise TypeError(
                "edge_index is not in the correct format."
            )
        if key == "edge_index":
            raise ValueError(
                "edge_index cannot be selected."
            )
        if key not in self.keys or not torch.is_tensor(self[key]):
            return None
        attributes = torch.index_select(self[key], 0, edge_index)
        if self.is_undirected():
            attributes = torch.cat([attributes, attributes], dim=0)
        return attributes

    def _update_graphs(self, verbose: bool = False):
        r"""
        Update the .G graph object with new attributes.
        The edges remain unchanged
        (edge_index should not be directly modified).

        The counter-part of update_tensors.
        """
        for key in self.keys:
            if Graph._is_node_attribute(key):
                Graph.add_node_attr(self.G, key, self[key])
            elif Graph._is_edge_attribute(key):
                # the order of edge attributes is consistent with edge index
                Graph.add_edge_attr(self.G, key, self[key])
            elif Graph._is_graph_attribute(key):
                Graph.add_graph_attr(self.G, key, self[key])
            else:
                if verbose:
                    print(f"Index fields: {key} ignored.")

    def apply_transform(
        self,
        transform,
        update_tensor: bool = True,
        update_graph: bool = False,
        deep_copy: bool = False,
        **kwargs
    ):
        r"""
        Applies the transformation function to current graph object.

        .. note::
            When the backend graph object (e.g. networkx object) is
            changed in the transform function, the argument :obj:`update_tensor` 
            is recommended, which will update the tensor representation to be in 
            sync with the transformed graph. Similarly, :obj:`update_graph` is 
            recommended when the transform function makes change to the tensor 
            objects.

            However, the transformation function should not make changes to both 
            of the backend graph object and the tensors simultaneously. Otherwise 
            there might exist inconsistency between the transformed graph and 
            tensors. Also note that :obj:`update_tensor` and :obj:`update_graph` 
            cannot be `True` at the same time.

            It is also possible to set both :obj:`update_tensor` and 
            :obj:`update_graph` to be `False`. This usually happens when one 
            needs to transform the tensor representation, but do not require that 
            the internal graph object to be in sync, for better efficiency. 
            In this case, the user should note that the internal `G` object is stale, 
            and that applying a transform in the future with 
            :obj:`update_tensor=True` will overwrite the current
            transformmation (which with parameters 
            :obj:`update_tensor=False; update_graph=False`).

        Args:
            transform (callable): In the format
                of :obj:`transform(deepsnap.graph.Graph, **kwargs)`.
                The function might need to return :class:`deepsnap.graph.Graph`
                (the transformed graph object).
            update_tensor (bool): If the graph has changed, use the 
                graph to update the stored tensor attributes.
            update_graph: (bool): If the tensor attributes have changed, 
                use the attributes to update the graph.
            deep_copy (bool): If `True`, the graph will be deepcopied 
                and then fed into the :meth:`transform` function.
                In this case, the :meth:`transform` function might also 
                need to return a :class:`Graph` object.

                .. note::
                    When returning :obj:`Graph` object in the transform function,
                    user should decide whether the tensor values of the graph is
                    to be copied (deepcopy).
            **kwargs (optional): Parameters used in the :meth:`transform` 
                function.

        Returns:
            :class:`Graph`: The transformed :class:`Graph` object.

        Note:
            This function is different from the function :meth:`apply_tensor`.
        """
        if update_tensor and update_graph:
            raise ValueError(
                "Tensor and graph should not be specified together."
            )
        graph_obj = copy.deepcopy(self) if deep_copy else self
        return_graph = transform(graph_obj, **kwargs)

        if isinstance(return_graph, self.__class__):
            return_graph = return_graph
        elif return_graph is None:
            # no return value; assumes in-place transform of the graph object
            return_graph = graph_obj
        else:
            raise TypeError(
                "Transform function returns a value of unknown type "
                f"({return_graph.__class__})"
            )
        if update_graph:
            if self.G is None:
                raise ValueError("There is no G in the class.")
            return_graph._update_graphs()
        if update_tensor:
            return_graph._update_tensors()
        return return_graph

    def apply_transform_multi(
        self,
        transform,
        update_tensors: bool = True,
        update_graphs: bool = False,
        deep_copy: bool = False,
        **kwargs
    ):
        r"""
        Applies transform function to the current graph object.
        But Unlike the :meth:`apply_transform`, the transform 
        argument in this method can return a tuple of graphs 
        (:class:`Graph`).

        Args:
            transform (callable): In the format of 
                :obj:`transform(deepsnap.graph.Graph, **kwargs)`.
                The function might need to return a tuple of graphs 
                that each has the type :class:`deepsnap.graph.Graph` 
                (the transformed graph objects).
            update_tensors (bool): If the graphs have changed, use the 
                graph to update the stored tensor attributes.
            update_graphs: (bool): If the tensor attributes have changed, 
                use the attributes to update the graphs.
            deep_copy (bool): If `True`, the graph will be deepcopied 
                and then fed into the :meth:`transform` function.
                In this case, the :meth:`transform` function might also 
                need to return a `Graph` object.
            **kwargs (optional): Parameters used in the :meth:`transform` 
                function.

        Returns:
            tuple: A tuple of transformed :class:`Graph` objects.
        """
        if update_tensors and update_graphs:
            raise ValueError(
                "Tensor and graph should not be specified together."
            )
        graph_obj = copy.deepcopy(self) if deep_copy else self
        return_graphs = transform(graph_obj, **kwargs)

        if isinstance(return_graphs[0], self.__class__):
            return_graphs = return_graphs
        elif return_graphs is None or len(return_graphs) == 0:
            # no return value; assumes in-place transform of the graph object
            return_graphs = (graph_obj,)
        else:
            raise TypeError(
                "Transform function returns a value of unknown type "
                f"({return_graphs[0].__class__})."
            )
        if update_graphs:
            for return_graph in return_graphs:
                if self.G is None:
                    raise ValueError("There is no G in the class.")
                return_graph._update_graphs()
        if update_tensors:
            for return_graph in return_graphs:
                return_graph._update_tensors()
        return return_graphs

    def _custom_split_node(self):
        r"""
        custom support version of _split_node
        """
        split_num = len(self.general_splits)
        split_graph = []

        for i in range(split_num):
            graph = copy.copy(self)
            graph.node_label_index = self._node_to_index(
                self.general_splits[i]
            )

            node_labels = []
            for node in self.general_splits[i]:
                node_label = node[-1]["node_label"]
                node_labels.append(node_label)
            node_labels = torch.tensor(node_labels)
            graph.node_label = node_labels
            split_graph.append(graph)
        return split_graph

    def _custom_split_edge(self):
        r"""
        custom support version of _split_edge
        """
        split_num = len(self.general_splits)
        split_graph = []

        for i in range(split_num):
            graph = copy.copy(self)
            graph.edge_label_index = self._edge_to_index(
                graph.general_splits[i]
            )

            edge_labels = []
            for edge in graph.general_splits[i]:
                edge_label = edge[-1]["edge_label"]
                edge_labels.append(edge_label)
            edge_labels = torch.tensor(edge_labels)

            if self.is_undirected():
                edge_labels = torch.cat(
                    [edge_labels, edge_labels]
                )
            graph.edge_label = edge_labels
            split_graph.append(graph)
        return split_graph

    def _custom_split(
        self,
        task: str
    ):
        r"""
        custom support version of split
        """
        if task == "node":
            return self._custom_split_node()
        elif task == "edge":
            return self._custom_split_edge()
        elif task == "link_pred":
            return self._custom_split_link_pred()
        elif task == "graph":
            raise ValueError("Graph task does not split individual graphs.")
        else:
            raise ValueError("Unknown task.")

    def split(
        self,
        task: str = "node",
        split_ratio: List[float] = None,
        shuffle: bool = True
    ):
        r"""
        Split current graph object to a list of graph objects.

        Args:
            task (str): One of `node`, `edge` or `link_pred`.
            split_ratio (list): A list of ratios such as
                `[train_ratio, validation_ratio, test_ratio]`. Default 
                is `[0.8, 0.1, 0.1]`.
            shuffle (bool): Whether to shuffle data for the splitting.

        Returns:
            list: A list of :class:`Graph` objects.
        """
        if split_ratio is None:
            split_ratio = [0.8, 0.1, 0.1]
        if not isinstance(split_ratio, list):
            raise TypeError("split ratio must be a list.")
        if len(split_ratio) > 3:
            raise ValueError("split ratio must contain leq three values")
        if not math.isclose(sum(split_ratio), 1.0):
            raise ValueError("split ratio must sum up to 1.")
        if not all(
            isinstance(split_ratio_i, float) for split_ratio_i in split_ratio
        ):
            raise TypeError("split ratio must contain all floats")
        if not all(split_ratio_i > 0 for split_ratio_i in split_ratio):
            raise ValueError("split ratio must contain all positivevalues.")

        if task == "node":
            return self._split_node(split_ratio, shuffle=shuffle)
        elif task == "edge":
            return self._split_edge(split_ratio, shuffle=shuffle)
        elif task == "link_pred":
            return self.split_link_pred(split_ratio, shuffle=shuffle)
        elif task == "graph":
            raise ValueError("Graph task does not split individual graphs.")
        else:
            raise ValueError("Unknown task.")

    def _split_node(self, split_ratio: float, shuffle: bool = True):
        r"""
        Split the graph into len(split_ratio) graphs for node prediction.
        Internally this splits node indices, and the model will only compute
        loss for the embedding of
        nodes in each split graph.
        In node classification, the whole graph is observed in train/val/test
        Only split over node_label_index
        """
        if self.num_nodes < len(split_ratio):
            raise ValueError(
                "In _split_node num of nodes are smaller than"
                "number of splitted parts."
            )

        split_graphs = []
        if shuffle:
            shuffled_node_indices = torch.randperm(self.num_nodes)
        else:
            shuffled_node_indices = torch.arange(self.num_nodes)

        # used to indicate whether default splitting results in
        # empty splitted graphs
        split_empty_flag = False
        nodes_split_list = []

        # perform `default split`
        split_offset = 0
        for i, split_ratio_i in enumerate(split_ratio):
            if i != len(split_ratio) - 1:
                num_split_i = int(split_ratio_i * self.num_nodes)
                nodes_split_i = shuffled_node_indices[
                    split_offset:split_offset + num_split_i
                ]
                split_offset += num_split_i
            else:
                nodes_split_i = shuffled_node_indices[split_offset:]

            if nodes_split_i.numel() == 0:
                split_empty_flag = True
                split_offset = 0
                nodes_split_list = []
                break
            nodes_split_list.append(nodes_split_i)

        if split_empty_flag:
            # perform `secure split` s.t. guarantees all splitted subgraph
            # contains at least one node.
            for i, split_ratio_i in enumerate(split_ratio):
                if i != len(split_ratio) - 1:
                    num_split_i = 1 + int(
                        split_ratio_i * (self.num_nodes - len(split_ratio))
                    )
                    nodes_split_i = shuffled_node_indices[
                        split_offset:split_offset + num_split_i
                    ]
                    split_offset += num_split_i
                else:
                    nodes_split_i = shuffled_node_indices[split_offset:]

                nodes_split_list.append(nodes_split_i)

        for nodes_split_i in nodes_split_list:
            # shallow copy all attributes
            graph_new = copy.copy(self)
            graph_new.node_label_index = nodes_split_i
            graph_new.node_label = self.node_label[nodes_split_i]
            split_graphs.append(graph_new)
        return split_graphs

    def _split_edge(self, split_ratio: float, shuffle: bool = True):
        r"""
        Split the graph into len(split_ratio) graphs for node prediction.
        Internally this splits node indices, and the model will only compute
        loss for the embedding of nodes in each split graph.
        In edge classification, the whole graph is observed in train/val/test.
        Only split over edge_label_index.
        """
        if self.num_edges < len(split_ratio):
            raise ValueError(
                "In _split_node num of edges are smaller than"
                "number of splitted parts."
            )

        split_graphs = []
        if shuffle:
            shuffled_edge_indices = torch.randperm(self.num_edges)
        else:
            shuffled_edge_indices = torch.arange(self.num_edges)
        split_offset = 0

        # used to indicate whether default splitting results in
        # empty splitted graphs
        split_empty_flag = False
        edges_split_list = []

        for i, split_ratio_i in enumerate(split_ratio):
            if i != len(split_ratio) - 1:
                num_split_i = int(split_ratio_i * self.num_edges)
                edges_split_i = shuffled_edge_indices[
                    split_offset:split_offset + num_split_i
                ]
                split_offset += num_split_i
            else:
                edges_split_i = shuffled_edge_indices[split_offset:]

            if edges_split_i.numel() == 0:
                split_empty_flag = True
                split_offset = 0
                edges_split_list = []
                break
            edges_split_list.append(edges_split_i)

        if split_empty_flag:
            # perform `secure split` s.t. guarantees all splitted subgraph
            # contains at least one edge.
            for i, split_ratio_i in enumerate(split_ratio):
                if i != len(split_ratio) - 1:
                    num_split_i = 1 + int(
                        split_ratio_i * (self.num_edges - len(split_ratio))
                    )
                    edges_split_i = shuffled_edge_indices[
                        split_offset:split_offset + num_split_i
                    ]
                    split_offset += num_split_i
                else:
                    edges_split_i = shuffled_edge_indices[split_offset:]
                edges_split_list.append(edges_split_i)

        for edges_split_i in edges_split_list:
            # shallow copy all attributes
            graph_new = copy.copy(self)
            graph_new.edge_label_index = self.edge_index[:, edges_split_i]
            graph_new.edge_label = self.edge_label[edges_split_i]
            split_graphs.append(graph_new)
        return split_graphs

    def _custom_split_link_pred_disjoint(self):
        r"""
        custom support version of disjoint split_link_pred
        """
        objective_edges = self.disjoint_split
        objective_edges_no_info = [edge[:-1] for edge in objective_edges]
        message_edges_no_info = (
            list(set(self.G.edges) - set(objective_edges_no_info))
        )
        if len(message_edges_no_info[0]) == 3:
            message_edges = [
                (
                    edge[0], edge[1], edge[2],
                    self.G.edges[edge[0], edge[1], edge[2]]
                )
                for edge in message_edges_no_info
            ]
        elif len(message_edges_no_info[0]) == 2:
            message_edges = [
                (edge[0], edge[1], self.G.edges[edge[0], edge[1]])
                for edge in message_edges_no_info
            ]
        else:
            raise ValueError("Each edge has more than 3 indices.")
        graph_train = Graph(
            self._edge_subgraph_with_isonodes(
                self.G,
                message_edges,
            )
        )
        graph_train.negative_label_val = self.negative_label_val
        graph_train._create_label_link_pred(
            graph_train, objective_edges
        )
        graph_train._is_train = True
        return graph_train

    def _custom_split_link_pred(self):
        r"""
        custom support version of split_link_pred
        """
        split_num = len(self.general_splits)
        split_graph = []

        edges_train = self.general_splits[0]
        edges_val = self.general_splits[1]

        graph_train = Graph(
            self._edge_subgraph_with_isonodes(
                self.G,
                edges_train,
            ),
            disjoint_split=(
                self.disjoint_split
            ),
            negative_edges=(
                self.negative_edges
            )
        )
        graph_train.negative_label_val = self.negative_label_val

        graph_val = copy.copy(graph_train)
        if split_num == 3:
            edges_test = self.general_splits[2]
            graph_test = Graph(
                self._edge_subgraph_with_isonodes(
                    self.G,
                    edges_train + edges_val
                ),
                negative_edges=(
                    self.negative_edges
                )
            )
            graph_test.negative_label_val = self.negative_label_val

        graph_train._create_label_link_pred(
            graph_train, edges_train
        )
        graph_val._create_label_link_pred(
            graph_val, edges_val
        )

        if split_num == 3:
            graph_test._create_label_link_pred(
                graph_test, edges_test
            )

        graph_train._is_train = True
        split_graph.append(graph_train)
        split_graph.append(graph_val)
        if split_num == 3:
            split_graph.append(graph_test)

        return split_graph

    def split_link_pred(
        self,
        split_ratio: Union[float, List[float]],
        shuffle: bool = True
    ):
        r"""
        Split the graph into `len(split_ratio)` graphs for 
        the link prediction task. Internally this function splits the edge indices, and 
        the model will only compute loss for the node embeddings in each splitted graph.
        This is only used for the transductive link prediction task.
        In this task, different parts of the graph are observed in train / val / test.
        If during training, we might further split the training graph for the
        message edges and supervision edges.

        .. note::

            This functon will be called twice.

        Args:
            split_ratio (float or list): The ratio or list of ratios.
            shuffle (bool): Whether to shuffle for the splitting.

        Returns:
            list: A list of :class:`Graph` objects.

        """
        if isinstance(split_ratio, float):
            split_ratio = [split_ratio, 1 - split_ratio]
        if len(split_ratio) < 2 or len(split_ratio) > 3:
            raise ValueError("Unrecoginzed number of splits")
        if self.num_edges < len(split_ratio):
            raise ValueError(
                "In _split_link_pred num of edges are smaller than"
                "number of splitted parts."
            )

        if self.G is not None:
            edges = list(self.G.edges(data=True))
            if shuffle:
                random.shuffle(edges)
        else:
            if shuffle:
                edges = torch.randperm(self.num_edges)
            else:
                edges = torch.arange(self.num_edges)

        # Perform `secure split` s.t. guarantees all splitted subgraph
        # that contains at least one edge.
        if len(split_ratio) == 2:
            num_edges_train = int(split_ratio[0] * self.num_edges)
            num_edges_val = self.num_edges - num_edges_train
            if (
                (num_edges_train == 0)
                or (num_edges_val == 0)
            ):
                num_edges_train = (
                    1 + int(split_ratio[0] * (self.num_edges - 2))
                )

            edges_train = edges[:num_edges_train]
            edges_val = edges[num_edges_train:]
        elif len(split_ratio) == 3:
            num_edges_train = int(split_ratio[0] * self.num_edges)
            num_edges_val = int(split_ratio[1] * self.num_edges)
            num_edges_test = self.num_edges - num_edges_train - num_edges_val
            if (
                (num_edges_train == 0)
                or (num_edges_val == 0)
                or (num_edges_test == 0)
            ):
                num_edges_train = (
                    1 + int(split_ratio[0] * (self.num_edges - 3))
                )
                num_edges_val = 1 + int(split_ratio[1] * (self.num_edges - 3))

            edges_train = edges[:num_edges_train]
            edges_val = edges[num_edges_train:num_edges_train + num_edges_val]
            edges_test = edges[num_edges_train + num_edges_val:]

        if self.G is not None:
            graph_train = Graph(
                self._edge_subgraph_with_isonodes(self.G, edges_train)
            )
            if hasattr(self, "negative_label_val"):
                graph_train.negative_label_val = self.negative_label_val
        else:
            graph_train = copy.copy(self)

            # update the edge_index
            edge_index = torch.index_select(
                self.edge_index, 1, edges_train
            )
            if self.is_undirected():
                edge_index = torch.cat(
                    [edge_index, torch.flip(edge_index, [0])], dim=1
                )

            # update edge features
            graph_train.edge_index = edge_index
            for key in graph_train.keys:
                if self._is_edge_attribute(key):
                    edge_feature = torch.index_select(
                        self[key], 0, edges_train
                    )
                    if self.is_undirected():
                        edge_feature = torch.cat(
                            [edge_feature, edge_feature], dim=0
                        )
                    graph_train[key] = edge_feature

            # in tensor backend, store the original self.edge_label
            graph_train._edge_label = copy.deepcopy(self.edge_label)

        graph_val = copy.copy(graph_train)
        if len(split_ratio) == 3:
            if self.G is not None:
                graph_test = Graph(
                    self._edge_subgraph_with_isonodes(
                        self.G, edges_train + edges_val
                    )
                )
                if hasattr(self, "negative_label_val"):
                    graph_test.negative_label_val = self.negative_label_val
            else:
                graph_test = copy.copy(self)
                edge_index = torch.index_select(
                    self.edge_index, 1, torch.cat([edges_train, edges_val])
                )
                if self.is_undirected():
                    edge_index = torch.cat(
                        [edge_index, torch.flip(edge_index, [0])],
                        dim=1
                    )
                graph_test.edge_index = edge_index
                for key in graph_test.keys:
                    if self._is_edge_attribute(key):
                        edge_feature = torch.index_select(
                            self[key], 0, torch.cat([edges_train, edges_val])
                        )
                        if self.is_undirected():
                            edge_feature = torch.cat(
                                [edge_feature, edge_feature], dim=0
                            )
                        graph_test[key] = edge_feature

        self._create_label_link_pred(graph_train, edges_train)
        self._create_label_link_pred(graph_val, edges_val)
        graph_train._is_train = True
        if len(split_ratio) == 3:
            self._create_label_link_pred(graph_test, edges_test)
            return [graph_train, graph_val, graph_test]
        else:
            return [graph_train, graph_val]

    def _edge_subgraph_with_isonodes(self, G, edges):
        r"""
        Generate a new networkx graph with same nodes and their attributes.

        Preserves all nodes and a subset of edges. Nodes that are not connected
        by any of the edges will be isolated nodes instead of being removed.

        Note:
            edges should be list(G_i.edges(data=True))
        """
        G_new = G.__class__()
        G_new.add_nodes_from(G.nodes(data=True))
        G_new.add_edges_from(edges)
        return G_new

    def resample_disjoint(self, message_ratio: float):
        r"""
        Resample splits of the message passing and supervision edges in the 
        `disjoint` mode.

        .. note::

            If :meth:`apply_transform` (on the message passing graph)
            was used before this resampling, it needs to be
            re-applied after resampling, to update some of the (supervision)
            edges that were in the objectives.

        Args:
            message_ratio(float): Split ratio.
        """
        if not hasattr(self, "_objective_edges"):
            raise ValueError("No disjoint edge split was performed.")

        # Combine into 1 graph
        if not hasattr(self, "_resample_disjoint_idx"):
            self._resample_disjoint_idx = 0

        resample_disjoint_period = self.resample_disjoint_period
        if self._resample_disjoint_idx == (resample_disjoint_period - 1):
            if self.G is not None:
                graph = self
                graph.G.add_edges_from(self._objective_edges)
            else:
                graph = copy.deepcopy(self)
                edge_index = graph.edge_index[:, 0:self.num_edges]
                # recover full edge_index
                edge_index = torch.cat(
                    [edge_index, graph._objective_edges], dim=1
                )
                if graph.is_undirected():
                    edge_index = torch.cat(
                        [edge_index, torch.flip(edge_index, [0])], dim=1
                    )
                graph.edge_index = edge_index

                # recover full edge attributes
                for key in graph._objective_edges_attribute:
                    if graph._is_edge_attribute(key):
                        graph[key] = torch.cat(
                            [
                                graph[key],
                                graph._objective_edges_attribute[key]
                            ],
                            dim=0
                        )
                        if graph.is_undirected():
                            graph[key] = torch.cat(
                                [graph[key], graph[key]], dim=0
                            )
                graph.edge_label = graph._edge_label

            graph = graph.split_link_pred(message_ratio)[1]
            graph._is_train = True
            graph._resample_disjoint_flag = True
        else:
            graph = self
            graph._resample_disjoint_flag = False

        graph._resample_disjoint_idx = (
            (self._resample_disjoint_idx + 1)
            % resample_disjoint_period
        )
        return graph

    def _create_label_link_pred(self, graph, edges):
        r"""
        Create edge label and the corresponding label_index (edges)
        for link prediction.

        Modifies the graph argument by setting the fields edge_label_index
        and edge_label.

        Notice when the graph is tensor backend, the edges are the
        indices of edges.
        """
        if self.G is not None:
            graph.edge_label_index = self._edge_to_index(edges)
            graph.edge_label = (
                self._get_edge_attributes_by_key(edges, "edge_label")
            )
            # Keep a copy of original edges (and their attributes)
            # for resampling the disjoint split
            # (message passing and objective links)
            graph._objective_edges = edges
        else:
            edge_label_index = torch.index_select(
                self.edge_index, 1, edges
            )

            # store objective edges
            graph._objective_edges = copy.deepcopy(edge_label_index)

            if self.is_undirected():
                edge_label_index = torch.cat(
                    [edge_label_index, torch.flip(edge_label_index, [0])],
                    dim=1
                )

            graph.edge_label_index = edge_label_index
            graph.edge_label = (
                self._get_edge_attributes_by_key_tensor(edges, "edge_label")
            )

            # store objective edge attributes
            objective_edges_attribute = {}
            for key in graph.keys:
                if self._is_edge_attribute(key) and (key != "edge_label"):
                    edge_attribute = torch.index_select(
                        self[key], 0, edges
                    )
                    objective_edges_attribute[key] = edge_attribute
            graph._objective_edges_attribute = objective_edges_attribute

    def _custom_create_neg_sampling(
        self, negative_sampling_ratio: float, resample: bool = False
    ):
        r"""
        custom support version of _create_neg_sampling where negative edges
        are provided as self.negative_edge

        Args:
            negative_sampling_ratio (float or int): ratio of negative sampling
                edges compared with the original edges.
            resample (boolean): whether should resample.

        """
        if resample and self._num_positive_examples is not None:
            self.edge_label_index = self.edge_label_index[
                :, :self._num_positive_examples
            ]

        num_pos_edges = self.edge_label_index.shape[-1]
        num_neg_edges = int(num_pos_edges * negative_sampling_ratio)

        if self.edge_index.size() == self.edge_label_index.size() and (
            torch.sum(self.edge_index - self.edge_label_index) == 0
        ):
            # (train in 'all' mode)
            edge_index_all = self.edge_index
        else:
            edge_index_all = (
                torch.cat([self.edge_index, self.edge_label_index], -1)
            )

        if len(edge_index_all) > 0:
            if not torch.is_tensor(self.negative_edge):
                negative_edges_length = len(self.negative_edge)
                if negative_edges_length < num_neg_edges:
                    multiplicity = math.ceil(
                        num_neg_edges / negative_edges_length
                    )
                    self.negative_edge = self.negative_edge * multiplicity
                    self.negative_edge = self.negative_edge[:num_neg_edges]

                self.negative_edge = torch.tensor(
                    list(zip(*self.negative_edge))
                )
            if not hasattr(self, "_negative_edge_idx"):
                self._negative_edge_idx = 0

            negative_edges = self.negative_edge
            negative_edges_length = negative_edges.shape[1]

            if self._negative_edge_idx + num_neg_edges > negative_edges_length:
                negative_edges_begin = (
                    negative_edges[:, self._negative_edge_idx:]
                )
                negative_edges_end = negative_edges[
                    :, :self._negative_edge_idx
                    + num_neg_edges - negative_edges_length
                ]
                negative_edges = torch.cat(
                    [negative_edges_begin, negative_edges_end], axis=1
                )
            else:
                negative_edges = negative_edges[
                    :, self._negative_edge_idx:
                    self._negative_edge_idx + num_neg_edges
                ]
            self._negative_edge_idx = (
                (self._negative_edge_idx + num_neg_edges)
                % negative_edges_length
            )
        else:
            return torch.tensor([], dtype=torch.long)

        if not resample:
            if self.edge_label is None:
                # if label is not yet specified, use all ones for positives
                positive_label = torch.ones(num_pos_edges, dtype=torch.long)
                # if label is not yet specified, use all zeros for positives
                negative_label = torch.zeros(num_neg_edges, dtype=torch.long)
            else:
                # if label is specified, use max(positive_label) + 1
                # for negative labels
                positive_label = self.edge_label
                negative_label_val = self.negative_label_val
                negative_label = (
                    negative_label_val
                    * torch.ones(num_neg_edges, dtype=torch.long)
                )
            self.edge_label = (
                torch.cat(
                    [positive_label, negative_label], -1
                ).type(torch.long)
            )

        # append to edge_label_index
        self.edge_label_index = (
            torch.cat([self.edge_label_index, negative_edges], -1)
        )
        self._num_positive_examples = num_pos_edges

    def _create_neg_sampling(
        self, negative_sampling_ratio: float, resample: bool = False
    ):
        r"""
        Create negative samples for link prediction,
        and changes the edge_label and edge_label_index accordingly
        (if already existed).

        Simplest link prediction has no label. It will be treated as
        binary classification.
        edge_label will be set to 1 for positives and 0 for negative examples.

        For link prediction that requires prediction of edge type,
        it will be a multi-class classification task.
        edge_label will be set to the (original label + 1) for positives
        and 0 for negative examples.
        Hence the number of prediction classes will be incremented by 1.
        In this case dataset.num_edge_labels should be called after split
        (which calls this function).

        Args:
            negative_sampling_ratio (float or int): ratio of negative sampling
                edges compared with the original edges.
            resample (boolean): whether should resample.
        """
        if resample and self._num_positive_examples is not None:
            # remove previous negative samples first
            # if self._num_positive_examples is None then
            # no previous sampling was done
            self.edge_label_index = self.edge_label_index[
                :, :self._num_positive_examples
            ]

        num_pos_edges = self.edge_label_index.shape[-1]
        num_neg_edges = int(num_pos_edges * negative_sampling_ratio)

        if self.edge_index.size() == self.edge_label_index.size() and (
            torch.sum(self.edge_index - self.edge_label_index) == 0
        ):
            # (train in 'all' mode)
            edge_index_all = self.edge_index
        else:
            edge_index_all = (
                torch.cat([self.edge_index, self.edge_label_index], -1)
            )

        # handle multigraph
        if hasattr(self, "_edge_index_all"):
            if not torch.equal(self._edge_index_all, edge_index_all):
                edge_index_all_unique = torch.unique(edge_index_all, dim=1)
            else:
                edge_index_all_unique = self._edge_index_all_unique
        else:
            edge_index_all_unique = torch.unique(edge_index_all, dim=1)
            self._edge_index_all = edge_index_all
            self._edge_index_all_unique = edge_index_all_unique

        negative_edges = self.negative_sampling(
            edge_index_all_unique, self.num_nodes, num_neg_edges
        )

        if not resample:
            if self.edge_label is None:
                # if label is not yet specified, use all ones for positives
                positive_label = torch.ones(num_pos_edges, dtype=torch.long)
                # if label is not yet specified, use all zeros for positives
                negative_label = torch.zeros(num_neg_edges, dtype=torch.long)
            else:
                positive_label = self.edge_label
                negative_label_val = self.negative_label_val
                negative_label = (
                    negative_label_val
                    * torch.ones(num_neg_edges, dtype=torch.long)
                )
            self.edge_label = (
                torch.cat(
                    [positive_label, negative_label], -1
                ).type(torch.long)
            )

        # append negative edges to edge_label_index
        self.edge_label_index = (
            torch.cat([self.edge_label_index, negative_edges], -1)
        )

        self._num_positive_examples = num_pos_edges

    @staticmethod
    def add_node_attr(G, attr_name: str, node_attr):
        r"""
        Add node attribute into a NetworkX graph. Assume that the
        `node_attr` ordering is the same as the node ordering in `G`.

        Args:
            G (NetworkX Graph): A NetworkX graph.
            attr_name (str): Name of the node attribute to set.
            node_attr (array_like): Corresponding node attributes.
        """
        # TODO: Better method here?
        node_list = list(G.nodes)
        attr_dict = dict(zip(node_list, node_attr))
        deepsnap._netlib.set_node_attributes(G, attr_dict, name=attr_name)

    @staticmethod
    def add_edge_attr(G, attr_name: str, edge_attr):
        r"""
        Add edge attribute into a NetworkX graph.

        Args:
            G (NetworkX Graph): A NetworkX graph.
            attr_name (str): Name of the edge attribute to set.
            edge_attr (array_like): Corresponding edge attributes.
        """
        # TODO: parallel?
        edge_list = list(G.edges)
        attr_dict = dict(zip(edge_list, edge_attr))
        deepsnap._netlib.set_edge_attributes(G, attr_dict, name=attr_name)

    @staticmethod
    def add_graph_attr(G, attr_name: str, graph_attr):
        r"""
        Add graph attribute into a NetworkX graph.

        Args:
            G (NetworkX Graph): A NetworkX graph.
            attr_name (str): Name of the graph attribute to set.
            graph_attr (scalar or array_like): Corresponding 
                graph attributes.
        """
        G.graph[attr_name] = graph_attr

    @staticmethod
    def pyg_to_graph(
        data,
        verbose: bool = False,
        fixed_split: bool = False,
        tensor_backend: bool = False,
        netlib=None
    ):
        r"""
        Transform a :class:`torch_geometric.data.Data` object to a 
        :class:`Graph` object.

        Args:
            data (:class:`torch_geometric.data.Data`): A 
                :class:`torch_geometric.data.Data` object that will be 
                transformed to a :class:`deepsnap.grpah.Graph` 
                object.
            verbose (bool): Whether to print information such as warnings.
            fixed_split (bool): Whether to load the fixed data split from 
                the original PyTorch Geometric data.
            tensor_backend (bool): `True` will use pure tensors for graphs.
            netlib (types.ModuleType, optional): The graph backend module. 
                Currently DeepSNAP supports the NetworkX and SnapX (for 
                SnapX only the undirected homogeneous graph) as the graph 
                backend. Default graph backend is the NetworkX.

        Returns:
            :class:`Graph`: A new DeepSNAP :class:`Graph` object.
        """
        # all fields in PyG Data object
        kwargs = {}
        kwargs["node_feature"] = data.x if "x" in data.keys else None
        kwargs["edge_feature"] = (
            data.edge_attr if "edge_attr" in data.keys else None
        )
        kwargs["node_label"], kwargs["edge_label"] = None, None
        kwargs["graph_feature"], kwargs["graph_label"] = None, None
        if kwargs["node_feature"] is not None and data.y.size(0) == kwargs[
            "node_feature"
        ].size(0):
            kwargs["node_label"] = data.y
        elif kwargs["edge_feature"] is not None and data.y.size(0) == kwargs[
            "edge_feature"
        ].size(0):
            kwargs["edge_label"] = data.y
        else:
            kwargs["graph_label"] = data.y

        if not tensor_backend:
            if netlib is not None:
                deepsnap._netlib = netlib
            if data.is_directed():
                G = deepsnap._netlib.DiGraph()
            else:
                G = deepsnap._netlib.Graph()
            G.add_nodes_from(range(data.num_nodes))
            G.add_edges_from(data.edge_index.T.tolist())
        else:
            attributes = {}
            if not data.is_directed():
                row, col = data.edge_index
                mask = row < col
                row, col = row[mask], col[mask]
                edge_index = torch.stack([row, col], dim=0)
                edge_index = torch.cat(
                    [edge_index, torch.flip(edge_index, [0])],
                    dim=1
                )
            else:
                edge_index = data.edge_index
            attributes["edge_index"] = edge_index

        # include other arguments that are in the kwargs of pyg data object
        keys_processed = ["x", "y", "edge_index", "edge_attr"]
        for key in data.keys:
            if key not in keys_processed:
                kwargs[key] = data[key]

        # we assume that edge-related and node-related features are defined
        # the same as in Graph._is_edge_attribute and Graph._is_node_attribute
        for key, value in kwargs.items():
            if value is None:
                continue
            if Graph._is_node_attribute(key):
                if not tensor_backend:
                    Graph.add_node_attr(G, key, value)
                else:
                    attributes[key] = value
            elif Graph._is_edge_attribute(key):
                # TODO: make sure the indices of edge attributes are same with edge_index
                if not tensor_backend:
                    # the order of edge attributes is consistent
                    # with edge index
                    Graph.add_edge_attr(G, key, value)
                else:
                    attributes[key] = value
            elif Graph._is_graph_attribute(key):
                if not tensor_backend:
                    Graph.add_graph_attr(G, key, value)
                else:
                    attributes[key] = value
            else:
                if verbose:
                    print(f"Index fields: {key} ignored.")

        if fixed_split:
            masks = ["train_mask", "val_mask", "test_mask"]
            if not tensor_backend:
                graph = Graph(G, netlib=netlib)
            else:
                graph = Graph(**attributes)
            if graph.edge_label is not None:
                graph.negative_label_val = torch.max(graph.edge_label) + 1

            graphs = []
            for mask in masks:
                if mask in kwargs:
                    graph_new = copy.copy(graph)
                    graph_new.node_label_index = (
                        torch.nonzero(data[mask]).squeeze()
                    )
                    graph_new.node_label = (
                        graph_new.node_label[graph_new.node_label_index]
                    )
                    graphs.append(graph_new)
            return graphs
        else:
            if not tensor_backend:
                return Graph(G, netlib=netlib)
            else:
                graph = Graph(**attributes)
            if graph.edge_label is not None:
                graph.negative_label_val = torch.max(graph.edge_label) + 1
            return graph

    @staticmethod
    def raw_to_graph(data):
        r"""
        Write other methods for user to import their own data format and
        make sure all attributes of G are scalar or :class:`torch.Tensor`.

        ``Not implemented``
        """
        raise NotImplementedError

    @staticmethod
    def negative_sampling(edge_index, num_nodes: int, num_neg_samples: int):
        r"""
        Samples random negative edges for a heterogeneous graph given
        by :attr:`edge_index`.

        Args:
            edge_index (LongTensor): The indices for edges.
            num_nodes (int): Number of nodes.
            num_neg_samples (int): The number of negative samples to
                return.

        Returns:
            :class:`torch.LongTensor`: The :attr:`edge_index` tensor 
            for negative edges.

        """
        num_neg_samples_available = min(
            num_neg_samples, num_nodes * num_nodes - edge_index.shape[1]
        )

        if num_neg_samples_available == 0:
            raise ValueError(
                "No negative samples could be generated for a complete graph."
            )

        rng = range(num_nodes ** 2)
        # idx = N * i + j
        idx = (edge_index[0] * num_nodes + edge_index[1]).to("cpu")

        perm = torch.tensor(random.sample(rng, num_neg_samples_available))
        mask = torch.from_numpy(np.isin(perm, idx)).to(torch.bool)
        rest = torch.nonzero(mask).view(-1)
        while rest.numel() > 0:  # pragma: no cover
            tmp = torch.tensor(random.sample(rng, rest.size(0)))
            mask = torch.from_numpy(np.isin(tmp, idx)).to(torch.bool)
            perm[rest] = tmp
            rest = rest[torch.nonzero(mask).view(-1)]

        row = perm // num_nodes
        col = perm % num_nodes
        neg_edge_index = torch.stack([row, col], dim=0).long()
        if num_neg_samples_available < num_neg_samples:
            multiplicity = math.ceil(
                num_neg_samples / num_neg_samples_available
            )
            neg_edge_index = torch.cat([neg_edge_index] * multiplicity, dim=1)
            neg_edge_index = neg_edge_index[:, :num_neg_samples]

        return neg_edge_index.to(edge_index.device)
