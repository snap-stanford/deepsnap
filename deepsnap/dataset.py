import copy
import math
import random
import networkx as nx
import numpy as np
import torch
from deepsnap.graph import Graph
from deepsnap.hetero_graph import HeteroGraph
import pdb
from typing import (
    List,
    Union,
)


class Generator(object):
    r"""
    Abstract class of on the fly generator used in dataset.
    It generates graphs on the fly to be fed into the model.
    """
    def __init__(self, sizes, size_prob=None, dataset_len=0):
        self.sizes = sizes
        if sizes is not None:
            if size_prob is None:
                self.size_prob = np.ones(len(sizes)) / len(sizes)
            else:
                self.size_prob = size_prob
        # by default length of generator dataset is 0
        self._len = dataset_len

    def __len__(self):
        return self._len

    def set_len(self, dataset_len):
        self._len = dataset_len

    def _get_size(self, size=None):
        if size is None:
            return np.random.choice(
                self.sizes, size=1, replace=True, p=self.size_prob
            )[0]
        else:
            return size

    @property
    def num_node_labels(self):
        return 0

    @property
    def num_nodes(self):
        return 0

    @property
    def num_edge_labels(self):
        return 0

    @property
    def num_edges(self):
        return 0

    @property
    def num_graph_labels(self):
        return 0

    def generate(self):
        r"""
        Overwrite in subclass. Generates and returns a Graph object
        """
        return Graph(nx.Graph())


class EnsembleGenerator(Generator):
    def __init__(self, generators, gen_prob=None, dataset_len=0):
        r"""
        A generator that is an ensemble of many generators.

        Args:
            generators: A list of Generators.
            prob: A list with the same length as generators.
                Specifies the probability of sampling from each generator.
                If None, uniformly sample a generator.
        """
        super(EnsembleGenerator, self).__init__(None, dataset_len=dataset_len)
        if gen_prob is None:
            self.gen_prob = np.ones(len(generators)) / len(generators)
        else:
            self.gen_prob = gen_prob
        self.generators = generators

    @property
    def num_node_labels(self):
        r"""
        Returns number of the node labels in the generated graphs.

        Returns:
            int: The number of node labels.
        """
        return max([gen.num_node_labels for gen in self.generators])

    @property
    def num_nodes(self):
        r"""
        Returns number of the nodes in each generated graphs.

        Returns:
            list: List of the number of nodes.
        """
        return [gen.num_nodes for gen in self.generators]

    @property
    def num_edge_labels(self):
        r"""
        Returns number of the edge labels in the generated graphs.

        Returns:
            int: The number of edge labels.
        """
        return max([gen.num_edge_labels for gen in self.generators])

    @property
    def num_edges(self):
        r"""
        Returns number of the edges in each generated graphs.

        Returns:
            list: List of the number of edges.
        """
        return [gen.num_edges for gen in self.generators]

    @property
    def num_graph_labels(self):
        r"""
        Returns number of the graph labels in the generated graphs.

        Returns:
            int: The number of graph labels.
        """
        return max([gen.num_graph_labels for gen in self.generators])

    def generate(self, **kwargs):
        r"""
        Generate a list of graphs.

        Returns:
            list: Generated list of :class:`deepsnap.graph.Graph` objects.
        """
        gen = np.random.choice(self.generators, 1, p=self.gen_prob)[0]
        return gen.generate(**kwargs)


class GraphDataset(object):
    r"""
        A plain python object modeling a list of Graph with various
        (optional) attributes.

        Args:
            TODO: update comments
            graphs (list): A list of Graph.
            task (str): Task this GraphDataset is used for
                (task = 'node' or 'edge' or 'link_pred' or 'graph').
            general_splits_mode (str): Whether to use (general_splits_mode =
                "random": split the graph randomly according to some ratio;
                or "custom": split the graph where all subgraphs are cutomized).
            disjoint_split_mode (str): Whether to use (disjoint_split_mode =
                "random": in the disjoint mode, split the train graph randomly according to some ratio;
                or "custom": in the disjoint mode, split the train graph where all subgraphs are cutomized).
            edge_negative_sampling_ratio (float): The number of negative samples compared
                to that of positive data.
            edge_message_ratio (float): The number of message-passing edges
                compared to that of training edge objectives.
            edge_train_mode (str): Whether to use (edge_train_mode = 'all':
                training edge objectives are the same as the message-passing edges;
                or 'disjoint': training edge objectives are different from message-passing edges;
                or 'train_only': training edge objectives are always the training set edges).
            minimum_node_per_graph (int): If the number of nodes of a graph is smaller than this,
                that graph will be filtered out.
            edge_split_mode (str): Whether to use (edge_split_mode =
                "exact": split the heterogeneous graph according to both ratio and split type;
                or "approximate": split the heterogeneous graph regardless of the split type).
            generator (:class:`deepsnap.dataset.Generator`): The dataset can be on-the-fly-generated.
                When using on the fly generator, the graphs = [] or None, and
                a generator(Generator) is provided, with an overwritten
                generate() method.
        """
    def __init__(
        self, graphs, task: str = "node",
        custom_split_graphs: List[Graph] = None,
        edge_negative_sampling_ratio: float = 1,
        edge_message_ratio: float = 0.8,
        edge_train_mode: str = "all",
        edge_split_mode: str = "exact",
        minimum_node_per_graph: int = 5,
        generator=None,
        resample_negatives=False,
        resample_disjoint=False,
        resample_disjoint_period=1,
        negative_label_val=None
    ):

        if graphs is not None:
            # make sure graphs is a list
            if not isinstance(graphs, list):
                graphs = [graphs]

            # support user input a list of nx.Graph instead of Graph
            for i, graph in enumerate(graphs):
                if isinstance(graph, nx.Graph):
                    graphs[i] = Graph(graph)

        # validity check for `task`
        if task not in ["node", "edge", "link_pred", "graph"]:
            raise ValueError(
                "`task` must be one of 'node', 'edge', 'link_pred' or 'graph'"
            )

        # validity check for `edge_train_mode`
        if edge_train_mode not in ["all", "disjoint"]:
            raise ValueError("`edge_train_mode` must be 'all' or 'disjoint'")

        # validity check for `edge_split_mode`
        if edge_split_mode not in ["exact", "approximate"]:
            raise ValueError(
                "`edge_split_mode` must be 'exact' or 'approximate'"
            )

        # parameter initialization
        self.graphs = graphs
        self.task = task
        self.custom_split_graphs = custom_split_graphs
        self.edge_message_ratio = edge_message_ratio
        self.edge_negative_sampling_ratio = edge_negative_sampling_ratio
        self.edge_train_mode = edge_train_mode
        self.edge_split_mode = edge_split_mode
        self.minimum_node_per_graph = minimum_node_per_graph
        self.resample_negatives = resample_negatives
        self.resample_disjoint = resample_disjoint
        self.resample_disjoint_period = resample_disjoint_period
        self.negative_label_val = negative_label_val
        self._split_types = None
        self._is_tensor = False

        # graphs preprocessing
        if graphs is None or len(graphs) == 0:
            if generator is None:
                raise ValueError("Graphs are None")
            else:
                # on-the-fly dataset
                self.generator = generator
                self.graphs = None
                self.otf_device = None
        elif generator is not None:
            raise ValueError(
                "Both graphs and on-the-fly generator are "
                "provided (only one should be provided."
            )
        else:
            # by default on-the-fly generator is not used.
            # when generator is not provide
            self.generator = None

            # TODO: refactor this piece of logic
            # filter graphs that are too small
            if self.minimum_node_per_graph > 0:
                graphs_filter = []
                for graph in self.graphs:
                    if isinstance(graph, Graph):
                        if isinstance(graph, HeteroGraph):
                            if (
                                sum(graph.num_nodes().values())
                                >= self.minimum_node_per_graph
                            ):
                                graphs_filter.append(graph)
                        else:
                            if graph.num_nodes >= self.minimum_node_per_graph:
                                graphs_filter.append(graph)
                    else:
                        raise TypeError(
                            "element in self.graphs of unexpected type"
                        )
                self.graphs = graphs_filter

            for graph in self.graphs:
                if not hasattr(graph, "_custom_update_flag"):
                    # assign task to graph
                    graph.task = self.task

                    # custom support
                    if isinstance(graph, Graph):
                        if isinstance(graph, HeteroGraph):
                            mapping = {
                                x: x
                                for x in range(sum(graph.num_nodes().values()))
                            }
                        else:
                            mapping = {x: x for x in range(graph.num_nodes)}
                    else:
                        raise TypeError(
                            "element in self.graphs of unexpected type"
                        )

                    graph._custom_update(mapping)

            # TODO: add checker to make sure negative_label_val is set up with
            #       other appropriate parameters
            if self.task == "link_pred":
                if self.negative_label_val is None:
                    negative_label_val = 0
                    for graph in self.graphs:
                        if (
                            hasattr(graph, "edge_label")
                            and (graph.edge_label is not None)
                            and (self.task == "link_pred")
                        ):
                            if isinstance(graph, Graph):
                                if isinstance(graph, HeteroGraph):
                                    for message_type in graph.edge_label:
                                        negative_label_val = max(
                                            negative_label_val,
                                            torch.max(
                                                graph.edge_label[message_type]
                                            ) + 1
                                        )
                                else:
                                    negative_label_val = max(
                                        negative_label_val,
                                        torch.max(graph.edge_label) + 1
                                    )
                            else:
                                raise TypeError(
                                    "element in self.graphs of unexpected type"
                                )

                    self.negative_label_val = negative_label_val

                for graph in self.graphs:
                    graph.negative_label_val = (
                        copy.deepcopy(negative_label_val)
                    )

            self._update_tensor_negative_edges()
            self._custom_mode_update()
        self._reset_cache()

    def _update_tensor_negative_edges(self):
        """
        Custom link prediction cases for homogeneous tensor backend
        """
        if self.task != "link_pred":
            return
        if not all([graph.G is None for graph in self.graphs]):
            return

        any_negative_edges = any(
            ["negative_edge" in graph.keys for graph in self.graphs]
        )
        all_negative_edges = all(
            ["negative_edge" in graph.keys for graph in self.graphs]
        )

        if (not all_negative_edges) and any_negative_edges:
            raise ValueError(
                "either all graphs have negative edges or no graphs have "
                "negative edges."
            )
        else:
            self._is_tensor = True
            for graph in self.graphs:
                graph._edge_label = copy.deepcopy(graph.edge_label)
                graph._edge_label_index = copy.deepcopy(graph.edge_label_index)
                if all_negative_edges:
                    graph._custom_create_neg_sampling(
                        self.edge_negative_sampling_ratio, resample=False
                    )
                else:
                    graph._create_neg_sampling(
                        self.edge_negative_sampling_ratio, resample=False
                    )

    def __len__(self) -> int:
        r"""
        Returns:
            int: The number of graph in graphs.
        """
        if self.graphs is None:
            return len(self.generator)
        else:
            return len(self.graphs)

    @property
    def num_node_features(self) -> int:
        r"""
        Returns node feature dimension in the graph.

        Returns:
            int: The number of features per node in the dataset.
        """
        return self._graph_example.num_node_features

    @property
    def num_node_labels(self) -> int:
        r"""
        Returns node feature dimension in the graph.

        Returns:
            int: The number of labels per node in the dataset.
        """
        if self._num_node_labels is None:
            if self.graphs is None:
                self._num_node_labels = self.generator.num_node_labels
            else:
                self._num_node_labels = (
                    max([graph.num_node_labels for graph in self.graphs])
                )
        return self._num_node_labels

    @property
    def num_nodes(self) -> List[int]:
        r"""
        Return number of nodes in graph list

        Returns:
            list: A list of number of nodes for each graph in graph list
        """
        if self._num_nodes is None:
            if self.graphs is None:
                self._num_nodes = self.generator.num_nodes
            else:
                self._num_nodes = (
                    [graph.num_nodes for graph in self.graphs]
                )
        return self._num_nodes

    @property
    def num_edge_features(self) -> int:
        r"""
        Returns edge feature dimension in the graph.

        Returns:
            int: The number of features per edge in the dataset.
        """
        return self._graph_example.num_edge_features

    @property
    def num_edge_labels(self) -> int:
        r"""
        Returns edge feature dimension in the graph.

        Returns:
            int: The number of labels per edge in the dataset.
        """
        if self._num_edge_labels is None:
            if self.graphs is None:
                self._num_edge_labels = self.generator.num_edge_labels
            else:
                self._num_edge_labels = (
                    max([graph.num_edge_labels for graph in self.graphs])
                )
        return self._num_edge_labels

    @property
    def num_edges(self) -> List[int]:
        r"""
        Return number of nodes in graph list

        Returns:
            list: A list of number of nodes for each graph in graph list
        """
        if self._num_edges is None:
            if self.graphs is None:
                self._num_edges = self.generator.num_edges
            else:
                self._num_edges = (
                    [graph.num_edges for graph in self.graphs]
                )
        return self._num_edges

    @property
    def num_graph_features(self) -> int:
        r"""
        Returns graph feature dimension in the graph.

        Returns:
            int: The number of features per graph in the dataset.
        """
        return self._graph_example.num_graph_features

    @property
    def num_graph_labels(self) -> int:
        r"""
        Returns graph feature dimension in the graph.

        Returns:
            int: The number of labels per graph in the dataset.
        """
        if self._num_graph_labels is None:
            if self.graphs is None:
                self._num_graph_labels = self.generator.num_graph_labels
            else:
                self._num_graph_labels = (
                    max([graph.num_graph_labels for graph in self.graphs])
                )
        return self._num_graph_labels

    @property
    def num_labels(self) -> int:
        r"""
        General wrapper that returns the number of labels depending on the task.

        Returns:
            int: The number of labels, depending on the task
        """
        if self.task == "node":
            return self.num_node_labels
        elif self.task == "edge" or self.task == "link_pred":
            return self.num_edge_labels
        elif self.task == "graph":
            return self.num_graph_labels
        else:
            raise ValueError(f"Task {self.task} not supported")

    def num_dims_dict(self):
        r"""
        Dimensions for all fields.

        Returns:
            dict: Name of the property to the dimension.
                e.g. 'node_feature' -> feature dimension;
                     'graph_label' -> label dimension
        """
        dim_dict = {}
        for key in self._graph_example.keys:
            tensor = self._graph_example[key]
            if not torch.is_tensor(tensor):
                continue
            if tensor.ndim == 1:
                dim_dict[key] = 1
            elif tensor.ndim == 2:
                dim_dict[key] = tensor.size()[-1]
            else:
                raise ValueError(f"Dimension of tensor {key} exceeds 2.")
        return dim_dict

    def _custom_mode_update(self):
        custom_keys = ["general_splits", "disjoint_split", "negative_edges"]
        for custom_key in custom_keys:
            self[f"{custom_key}_mode"] = "random"

        # make sure custom appeared in all graphs or no custom appeared in any graphs
        custom_in_graphs = all(
            (graph.custom is not None) for graph in self.graphs
        )
        custom_not_in_graphs = all(
            (graph.custom is None) for graph in self.graphs
        )
        if not custom_in_graphs and not custom_not_in_graphs:
            raise ValueError(
                "custom needs to be in all graphs or not in any graphs"
            )
        if custom_in_graphs:
            for custom_key in custom_keys:
                custom_key_in_custom = all(
                    (custom_key in graph.custom)
                    for graph in self.graphs
                )
                custom_key_not_in_custom = all(
                    (custom_key not in graph.custom) for graph in self.graphs
                )
                if not custom_key_in_custom and not custom_key_not_in_custom:
                    raise ValueError(
                        f"{custom_key} needs to be in all `graph.custom`s or "
                        "not in any `graph.custom`s"
                    )
                if custom_key_in_custom:
                    self[f"{custom_key}_mode"] = "custom"

        # custom inductive splits
        if self.custom_split_graphs is not None:
            self.general_splits_mode = "custom"

    def _split_transductive(
        self,
        split_ratio: List[float],
        split_types: List[str] = None,
        shuffle: bool = True
    ) -> List[Graph]:
        r"""
        Split the dataset assuming training process is transductive.

        Args:
            split_ratio: number of data splitted into train, validation
                (and test) set.

        Returns:
            list: A list of 3 (2) lists of :class:`deepsnap.graph.Graph` object corresponding
            to train, validation (and test) set.
        """
        if self.task == "graph":
            raise ValueError('Graph prediction task cannot be transductive')

        # a list of split graphs
        # (e.g. [[train graph, val graph, test graph], ... ])
        if self.general_splits_mode == "custom":
            split_num = len(self.graphs[0].general_splits)
            split_graphs = [[] for x in range(split_num)]
            # TODO: add _custom_split()
            for graph in self.graphs:
                if self.task == "link_pred":
                    split_graph = graph._custom_split_link_pred()
                    for i in range(split_num):
                        split_graphs[i].append(split_graph[i])
                if self.task == "node":
                    # TODO: add _custom_split_node()
                    for i in range(split_num):
                        if isinstance(graph, Graph):
                            graph_temp = copy.copy(graph)
                            graph_temp.node_label_index = (
                                graph._node_to_index(
                                    graph.general_splits[i]
                                )
                            )

                            if isinstance(graph, HeteroGraph):
                                node_labels = {}
                                for node in graph.general_splits[i]:
                                    node_label = node[-1]["node_label"]
                                    node_type = node[-1]["node_type"]
                                    if node_type not in node_labels:
                                        node_labels[node_type] = []
                                    node_labels[node_type].append(node_label)

                                for node_type in node_labels:
                                    node_labels[node_type] = torch.tensor(
                                        node_labels[node_type]
                                    )
                            else:
                                node_labels = []
                                for node in graph.general_splits[i]:
                                    node_label = node[-1]["node_label"]
                                    node_labels.append(node_label)
                                node_labels = torch.tensor(node_labels)

                            graph_temp.node_label = node_labels
                            split_graphs[i].append(graph_temp)
                        else:
                            raise TypeError(
                                "element in self.graphs of unexpected type."
                            )

                if self.task == "edge":
                    # TODO: add _custom_split_edge()
                    for i in range(split_num):
                        graph_temp = copy.copy(graph)
                        if isinstance(graph, Graph):
                            if isinstance(graph, HeteroGraph):
                                graph_temp.edge_label_index = (
                                    graph._edge_to_index(
                                        graph.general_splits[i],
                                        list(graph_temp.G.nodes(data=True))
                                    )
                                )
                                # create new edge_label accordingly
                                edge_labels = {}
                                for edge in graph.general_splits[i]:
                                    edge_type = edge[-1]["edge_type"]
                                    edge_label = edge[-1]["edge_label"]
                                    head_type = (
                                        graph.G.nodes[edge[0]]["node_type"]
                                    )
                                    tail_type = (
                                        graph.G.nodes[edge[1]]["node_type"]
                                    )
                                    message_type = (
                                        head_type, edge_type, tail_type
                                    )
                                    if message_type not in edge_labels:
                                        edge_labels[message_type] = []
                                    edge_labels[message_type].append(
                                        edge_label
                                    )
                                for message_type in edge_labels:
                                    edge_labels[message_type] = torch.tensor(
                                        edge_labels[message_type]
                                    )
                                if graph.is_undirected():
                                    for message_type in edge_labels:
                                        edge_labels[message_type] = torch.cat(
                                            [
                                                edge_labels[message_type],
                                                edge_labels[message_type]
                                            ],
                                            dim=1
                                        )
                            else:
                                graph_temp.edge_label_index = (
                                    graph._edge_to_index(
                                        graph.general_splits[i]
                                    )
                                )
                                # create new edge_label accordingly
                                edge_labels = []
                                for edge in graph.general_splits[i]:
                                    edge_label = edge[-1]["edge_label"]
                                    edge_labels.append(edge_label)
                                edge_labels = torch.tensor(edge_labels)
                                if graph.is_undirected():
                                    edge_labels = torch.cat(
                                        [edge_labels, edge_labels], dim=1
                                    )

                            graph_temp.edge_label = edge_labels
                            split_graphs[i].append(graph_temp)
                        else:
                            raise TypeError(
                                "element in self.graphs of unexpected type."
                            )

        # TODO: add checker to make sure edge_split_mode
        elif self.general_splits_mode == "random":
            split_graphs = []
            for graph in self.graphs:
                if isinstance(graph, Graph):
                    if isinstance(graph, HeteroGraph):
                        split_graph = graph.split(
                            task=self.task,
                            split_types=split_types,
                            split_ratio=split_ratio,
                            edge_split_mode=self.edge_split_mode,
                            shuffle=shuffle
                        )
                    else:
                        split_graph = graph.split(
                            self.task, split_ratio, shuffle=shuffle
                        )
                else:
                    raise TypeError(
                        "element in self.graphs of unexpected type"
                    )
                split_graphs.append(split_graph)
            split_graphs = list(map(list, zip(*split_graphs)))

        # TODO: reorg these checkers
        if self.disjoint_split_mode == "custom":
            # resample_disjoint when in disjoint split custom mode
            # would override the custom disjoint split edges
            self.resample_disjoint = False
            for i, graph in enumerate(split_graphs[0]):
                if (
                    self.task == "link_pred"
                    and self.edge_train_mode == "disjoint"
                ):
                    graph = graph._custom_split_link_pred_disjoint()
                    split_graphs[0][i] = graph

        elif self.disjoint_split_mode == "random":
            for i, graph in enumerate(split_graphs[0]):
                if (
                    self.task == "link_pred"
                    and self.edge_train_mode == "disjoint"
                ):
                    if isinstance(graph, Graph):
                        # store the original edge_label
                        graph_edge_label = None
                        if (
                            self.resample_disjoint
                            and hasattr(graph, "edge_label")
                        ):
                            graph_edge_label = graph.edge_label

                        if isinstance(graph, HeteroGraph):
                            graph = graph.split_link_pred(
                                split_types=split_types,
                                split_ratio=self.edge_message_ratio,
                                edge_split_mode=self.edge_split_mode
                            )[1]
                        else:
                            graph = graph.split_link_pred(
                                self.edge_message_ratio
                            )[1]
                        graph.is_train = True
                        split_graphs[0][i] = graph

                        # save the original edge_label
                        if graph_edge_label is not None:
                            graph._edge_label = copy.deepcopy(graph_edge_label)
                        else:
                            graph._edge_label = None
                    else:
                        raise TypeError(
                            "element in self.graphs of unexpected type"
                        )

        # list of num_splits datasets
        # (e.g. [train dataset, val dataset, test dataset])
        dataset_return = []
        if self.negative_edges_mode == "random":
            for x in split_graphs:
                dataset_current = copy.copy(self)
                dataset_current.graphs = x
                if self.task == "link_pred":
                    for graph_temp in dataset_current.graphs:
                        if isinstance(graph_temp, Graph):
                            if isinstance(graph_temp, HeteroGraph):
                                graph_temp._create_neg_sampling(
                                    negative_sampling_ratio=(
                                        self.edge_negative_sampling_ratio
                                    ),
                                    split_types=split_types
                                )
                            else:
                                graph_temp._create_neg_sampling(
                                    self.edge_negative_sampling_ratio
                                )
                        else:
                            raise TypeError(
                                "element in self.graphs of unexpected type"
                            )
                dataset_return.append(dataset_current)
        elif self.negative_edges_mode == "custom":
            for i, x in enumerate(split_graphs):
                dataset_current = copy.copy(self)
                dataset_current.graphs = x
                if self.task == "link_pred":
                    for j, graph_temp in enumerate(dataset_current.graphs):
                        if isinstance(graph_temp, Graph):
                            if isinstance(graph_temp, HeteroGraph):
                                graph_temp.negative_edge = (
                                    graph_temp.negative_edges[i]
                                )
                                graph_temp._custom_create_neg_sampling(
                                    self.edge_negative_sampling_ratio,
                                    split_types=split_types
                                )
                            else:
                                graph_temp.negative_edge = (
                                    graph_temp.negative_edges[i]
                                )
                                graph_temp._custom_create_neg_sampling(
                                    self.edge_negative_sampling_ratio
                                )
                        else:
                            raise TypeError(
                                "element in self.graphs of unexpected type"
                            )
                dataset_return.append(dataset_current)
        # resample negatives for train split (only for link prediction)
        dataset_return[0].resample_negatives = True
        return dataset_return

    def _split_inductive(
        self,
        split_ratio: List[float],
        split_types: List[str] = None,
        shuffle: bool = True
    ) -> List[Graph]:
        r"""
        Split the dataset assuming training process is inductive.

        Args:
            split_ratio: number of data splitted into train, validation
                (and test) set.

        Returns:
            List[Graph]: a list of 3 (2) lists of graph object corresponding to train, validation (and test) set.
        """
        if self.general_splits_mode == "custom":
            split_graphs = self.custom_split_graphs
        elif self.general_splits_mode == "random":
            num_graphs = len(self.graphs)
            if num_graphs < len(split_ratio):
                raise ValueError(
                    "in _split_inductive num of graphs are smaller than the "
                    "number of splitted parts"
                )

            if shuffle:
                self._shuffle()
            # a list of num_splits list of graphs
            # (e.g. [train graphs, val graphs, test graphs])
            split_graphs = []
            split_offset = 0

            # perform `secure split` s.t. guarantees all splitted graph list
            # contains at least one graph.
            for i, split_ratio_i in enumerate(split_ratio):
                if i != len(split_ratio) - 1:
                    num_split_i = (
                        1 +
                        int(split_ratio_i * (num_graphs - len(split_ratio)))
                    )
                    split_graphs.append(
                        self.graphs[split_offset: split_offset + num_split_i])
                    split_offset += num_split_i
                else:
                    split_graphs.append(self.graphs[split_offset:])

        # TODO: refactor this part of the code: split_graphs[i][j] -> graph
        # create objectives for link_pred task
        if self.task == "link_pred":
            # if disjoint, this will split all graph's edges into 2:
            # message passing and objective edges
            # which is returned by the [1] of the split graphs
            if self.edge_train_mode == "disjoint":
                split_start = 0
            # in all mode, train graph has all edges used for both
            # message passing and objective.
            elif self.edge_train_mode == "all":
                split_start = 1
            for i in range(split_start, len(split_graphs)):
                for j in range(len(split_graphs[i])):
                    if isinstance(split_graphs[i][j], Graph):
                        # store the original edge_label
                        graph_edge_label = None
                        if (
                            self.resample_disjoint
                            and (i == 0)
                            and hasattr(split_graphs[i][j], "edge_label")
                        ):
                            graph_edge_label = split_graphs[i][j].edge_label

                        if isinstance(split_graphs[i][j], HeteroGraph):
                            split_graphs[i][j] = (
                                split_graphs[i][j].split_link_pred(
                                    split_types,
                                    self.edge_message_ratio,
                                    self.edge_split_mode
                                )[1]
                            )
                        else:
                            split_graphs[i][j] = (
                                split_graphs[i][j].split_link_pred(
                                    self.edge_message_ratio
                                )[1]
                            )

                        # save the original edge_label
                        if graph_edge_label is not None:
                            split_graphs[i][j]._edge_label = (
                                copy.deepcopy(graph_edge_label)
                            )
                        else:
                            split_graphs[i][j]._edge_label = None
                    else:
                        raise TypeError(
                            "element in self.graphs of unexpected type."
                        )
                    # set is_train flag
                    if i == 0:
                        split_graphs[i][j].is_train = True

        # list of num_splits datasets
        dataset_return = []
        for graphs in split_graphs:
            dataset_current = copy.copy(self)
            dataset_current.graphs = graphs
            if self.task == "link_pred":
                for graph_temp in dataset_current.graphs:
                    if isinstance(graph_temp, Graph):
                        if isinstance(graph_temp, HeteroGraph):
                            graph_temp._create_neg_sampling(
                                negative_sampling_ratio=(
                                    self.edge_negative_sampling_ratio
                                ),
                                split_types=split_types
                            )
                        else:
                            graph_temp._create_neg_sampling(
                                self.edge_negative_sampling_ratio
                            )
                    else:
                        raise TypeError(
                            "element in self.graphs of unexpected type"
                        )
            dataset_return.append(dataset_current)

        # resample negatives for train split (only for link prediction)
        dataset_return[0].resample_negatives = True

        return dataset_return

    def split(
        self,
        transductive: bool = True,
        split_ratio: List[float] = None,
        split_types: Union[str, List[str]] = None,
        shuffle: bool = True
    ) -> List[Graph]:
        r""" Split datasets into train, validation (and test) set.

        Args:
            transductive: whether the training process is transductive
                or inductive. Inductive split is always used for graph-level tasks (
                self.task == 'graph').
            split_ratio: number of data splitted into train, validation
                (and test) set.

        Returns:
            list: a list of 3 (2) lists of :class:`deepsnap.graph.Graph` objects corresponding to train, validation (and test) set.
        """
        if self.graphs is None:
            raise RuntimeError(
                "Split is not supported for on-the-fly dataset. "
                "Construct different on-the-fly datasets for train, val "
                "and test. Or perform split at batch level."
            )
        if split_ratio is None:
            split_ratio = [0.8, 0.1, 0.1]
        if not isinstance(split_ratio, list):
            raise TypeError("Split ratio must be a list.")
        if len(split_ratio) > 3:
            raise ValueError(
                "Split ratio must contain less than or equal to three values."
            )
        if not math.isclose(sum(split_ratio), 1.0):
            raise ValueError("Split ratio must sum up to 1.")
        if not all(
            isinstance(split_ratio_i, float)
            for split_ratio_i in split_ratio
        ):
            raise TypeError("Split ratio must contain all floats.")
        if not all(split_ratio_i > 0 for split_ratio_i in split_ratio):
            raise ValueError("Split ratio must contain all positivevalues.")

        # store the most recent split types
        self._split_types = split_types

        # check self._is_tensor
        if self._is_tensor:
            for graph in self.graphs:
                graph.edge_label_index = graph._edge_label_index
                graph.edge_label = graph._edge_label

        # list of num_splits datasets
        dataset_return = []
        if transductive:
            if self.task == "graph":
                raise ValueError(
                    "in transductive mode, self.task is graph does not "
                    "make sense."
                )
            dataset_return = (
                self._split_transductive(
                    split_ratio, split_types, shuffle=shuffle
                )
            )
        else:
            dataset_return = (
                self._split_inductive(
                    split_ratio,
                    split_types,
                    shuffle=shuffle
                )
            )

        return dataset_return

    def resample_disjoint(self):
        r""" Resample disjoint edge split of message passing and objective links.

        Note that if apply_transform (on the message passing graph)
        was used before this resampling, it needs to be
        re-applied, after resampling, to update some of the edges that were in objectives.
        """
        if self.graphs is None:
            raise RuntimeError(
                "Resampling disjoint is not needed for on-the-fly dataset. "
                "Split the on-the-fly data as the batch arrives."
            )
        graphs = []
        for graph in self.graphs:
            graphs.append(graph.resample_disjoint(self.edge_message_ratio))
        self.graphs = graphs

    def _reset_cache(self):
        r"""
        Resets internal cache for graph examples, num_node_labels etc.
        """
        self._num_node_labels = None
        self._num_nodes = None
        self._num_edge_labels = None
        self._num_edges = None
        self._num_graph_labels = None
        # TODO: consider the heterogeneous graph case
        if self.graphs is None:
            self._graph_example = self.generator.generate()
            if not isinstance(self._graph_example, Graph):
                self._graph_example = Graph(self._graph_example)
        else:
            self._graph_example = self.graphs[0]

    def apply_transform(
        self, transform,
        update_tensor: bool = True,
        update_graph: bool = False,
        deep_copy: bool = False,
        **kwargs
    ):
        r"""
        Applies a transformation to each graph object in parallel by first
        calling `to_data_list`, applying the transform, and then perform
        re-batching again to a `GraphDataset`.

        Args:
            transform: user-defined transformation function.
            update_tensor: whether request the Graph object remain unchanged.
            kwargs: parameters used in transform function in Graph object.
        """
        # currently does not support transform for on-the-fly dataset
        if self.graphs is None:
            raise ValueError(
                "On-the-fly datasets do not support transform. "
                "Transform can be done at the batch level."
            )
        # TODO: parallel apply
        new_dataset = copy.copy(self)
        new_dataset.graphs = [
            graph.apply_transform(
                transform, update_tensor, update_graph,
                deep_copy, **kwargs
            )
            for graph in self.graphs
        ]
        # update example graph used for num_node_features etc.
        new_dataset._reset_cache()
        return new_dataset

    def filter(self, filter_fn, deep_copy: bool = False, **kwargs):
        r""" Filter the dataset, discarding graph data G where filter_fn(G) is False.

        GraphDataset.apply_transform is an analog of python map in graph dataset, while
        GraphDataset.filter is an analog of python filter.

        Args:
            filter_fn: user-defined filter function that returns True (keep) or
                False (discard) for graph object in this dataset.
            deep_copy: whether to deep copy all graph objects in the returned list.
            kwargs: parameters used in the filter function.

        Returns:
            A new dataset where graphs are filtered by the given filter function.
        """
        # currently does not support filter for on-the-fly dataset
        if self.graphs is None:
            raise ValueError(
                "On-the-fly datasets do not support transform."
                "Filter can be done at the batch level."
            )
        new_dataset = copy.copy(self)
        new_dataset.graphs = [
            graph for graph in self.graphs if filter_fn(graph, **kwargs)]
        # update example graph used for num_node_features etc.
        new_dataset._reset_cache()
        return new_dataset

    def to(self, device):
        r"""
        Transfer Graph object in the graphs to specified device.

        Args:
            device: Specified device name
        """
        if self.graphs is None:
            self.otf_device = device
        else:
            for graph in self.graphs:
                graph.to(device)

    def _shuffle(self):
        r"""
        shuffle Graph object in graphs.
        """
        if self.graphs is not None:
            random.shuffle(self.graphs)

    @staticmethod
    def pyg_to_graphs(
        dataset,
        verbose: bool = False,
        fixed_split: bool = False,
        tensor_backend: bool = False
    ) -> List[Graph]:
        r"""
        Transform a torch_geometric.data.Dataset object to a list of Graph object.

        Args:
            dataset: a torch_geometric.data.Dataset object.
            verbose: if print verbose warning
            fixed_split: if load fixed data split from PyG dataset
            tensor_backend: if using the tensor backend

        Returns:
            list: A list of :class:`deepsnap.graph.Graph` object.
        """
        if fixed_split:
            graphs = [
                Graph.pyg_to_graph(
                    data, verbose=verbose, fixed_split=True,
                    tensor_backend=tensor_backend
                )
                for data in dataset
            ]
            graphs_split = [[graph] for graph in graphs[0]]
            return graphs_split
        else:
            return [
                Graph.pyg_to_graph(
                    data, verbose=verbose,
                    tensor_backend=tensor_backend
                )
                for data in dataset
            ]

    def __getitem__(self, idx: int) -> Union[Graph, List[Graph]]:
        r"""
        Takes in an integer (or a list of integers)
        returns a single Graph object (a subset of graphs).

        Args:
            idx: index to be selected from graphs.

        Returns:
            Union[:class:`deepsnap.graph.Graph`, List[:class:`deepsnap.graph.Graph`]]: A single
            :class:`deepsnap.graph.Graph` object or subset of :class:`deepsnap.graph.Graph` objects.
        """
        # TODO: add the hetero graph equivalent of these functions ?
        if self.graphs is None:
            graph = self.generator.generate()
            if not isinstance(graph, Graph):
                graph = Graph(graph)
            # generated an networkx graph
            if self.otf_device is not None:
                graph.to(self.otf_device)
        elif isinstance(idx, int):
            graph = self.graphs[idx]
        else:
            # sliceing of dataset
            dataset = self._index_select(idx)
            return dataset


        # TODO: remove self.task check by add corresponding checker in the __init__
        if (
            self.task == "link_pred"
            and self.edge_train_mode == "disjoint"
            and self.resample_disjoint
            and graph.is_train
        ):
            if not hasattr(graph, "resample_disjoint_period"):
                graph.resample_disjoint_period = self.resample_disjoint_period

            if isinstance(graph, Graph):
                if isinstance(graph, HeteroGraph):
                    graph = graph.resample_disjoint(
                        split_types=self._split_types,
                        message_ratio=self.edge_message_ratio
                    )
                else:
                    graph = graph.resample_disjoint(self.edge_message_ratio)
            else:
                raise TypeError(
                    "element in self.graphs of unexpected type."
                )

        if self.task == "link_pred" and self.resample_negatives:
            resample_negative_flag = True
            # after graph just resampled disjoint training data
            # graph.edge_label is reset to original state,
            # the negative sampling process needs to utilize
            # resample=False mode.
            if (
                hasattr(graph, "_resample_disjoint_flag")
                and graph._resample_disjoint_flag
            ):
                resample_negative_flag = False

            # resample negative examples
            if isinstance(graph, Graph):
                if isinstance(graph, HeteroGraph):
                    if self.negative_edges_mode == "random":
                        graph._create_neg_sampling(
                            self.edge_negative_sampling_ratio,
                            split_types=self._split_types,
                            resample=resample_negative_flag
                        )
                    elif self.negative_edges_mode == "custom":
                        graph._custom_create_neg_sampling(
                            self.edge_negative_sampling_ratio,
                            split_types=self._split_types,
                            resample=resample_negative_flag
                        )
                else:
                    if self.negative_edges_mode == "random":
                        graph._create_neg_sampling(
                            self.edge_negative_sampling_ratio,
                            resample=resample_negative_flag
                        )
                    elif self.negative_edges_mode == "custom":
                        graph._custom_create_neg_sampling(
                            self.edge_negative_sampling_ratio,
                            resample=resample_negative_flag
                        )
            else:
                raise TypeError(
                    "element in self.graphs of unexpected type."
                )

        if self.graphs is not None and isinstance(idx, int):
            self.graphs[idx] = graph

        return graph

    def __setitem__(self, key: str, value):
        """Sets the attribute :obj:`key` to :obj:`value`."""
        setattr(self, key, value)

    def _index_select(self, idx: int) -> List[Graph]:
        r"""
        Takes in a list of integers, returns a subset of graphs
        corresponding to the list of integers.

        Args:
            idx: index to be selected from graphs.

        Returns:
            List[Graph]: a single Graph object or subset of graphs.
        """
        if self.graphs is None:
            # _index_select is only called when self.graphs is not None
            raise NotImplementedError(
                "Index select is not available for on-the-fly dataset."
            )

        if isinstance(idx, slice):
            dataset = copy.copy(self)
            dataset.graphs = self.graphs[idx]
        elif torch.is_tensor(idx):
            if (
                idx.dtype == torch.long
                or idx.dtype == torch.int
            ):
                dataset = self._index_select(idx.tolist())
            elif idx.dtype == torch.bool:
                dataset = self._index_select(idx.nonzero().flatten().tolist())
            else:
                raise TypeError(
                    f"your index type is {idx.dtype}, only tensor of type "
                    "torch.long, torch.int or torch.bool are accepted."
                )
        elif isinstance(idx, list) or isinstance(idx, tuple):
            dataset = copy.copy(self)
            dataset.graphs = [self.graphs[x] for x in idx]
        else:
            raise IndexError(
                "Only integers, slices (`:`), list, tuples, and long or bool "
                f"tensors are valid indices (got {type(idx).__name__})."
            )
        return dataset

    def __repr__(self) -> str:  # pragma: no cover
        descriptor = (
            len(self) if self.graphs is not None else self.generator.__class__
        )
        return f"{self.__class__.__name__}({descriptor})"
