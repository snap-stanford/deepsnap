import copy
import math
import types
import random
import networkx as nx
import numpy as np
import torch
import deepsnap
from deepsnap.graph import Graph
from deepsnap.hetero_graph import HeteroGraph
import pdb
from typing import (
    Dict,
    List,
    Union
)
import warnings


class Generator(object):
    r"""
    Abstract class of on the fly generator used in the dataset.
    It generates on the fly graphs, which will be fed into the model.
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
        Overwrite in subclass. Generates and returns a 
        :class:`deepsnap.graph.Graph` object

        Returns:
            :class:`deepsnap.graph.Graph`: A DeepSNAP graph object.
        """
        return Graph(nx.Graph())


class EnsembleGenerator(Generator):
    def __init__(self, generators, gen_prob=None, dataset_len=0):
        r"""
        A generator that is an ensemble of many generators.

        Args:
            generators (List[:class:`Generator`]): A list of Generators.
            gen_prob (Array like): An array like (list) probabilities with 
                the same length as generators. It specifies the probability 
                of sampling from each generator. If it is `None`, the 
                :class:`EnsembleGenerator` will uniformly sample a generator.
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
        # TODO: change to unique as what we did in graph.py
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
        # TODO: change to unique as what we did in graph.py
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
        # TODO: change to unique as what we did in graph.py
        return max([gen.num_graph_labels for gen in self.generators])

    def generate(self, **kwargs):
        r"""
        Generate a list of graphs.

        Returns:
            list: Generated a list of :class:`deepsnap.graph.Graph` objects.
        """
        gen = np.random.choice(self.generators, 1, p=self.gen_prob)[0]
        return gen.generate(**kwargs)


class GraphDataset(object):
    r"""
        A plain python object modeling a list of :class:`deepsnap.graph.Graph` 
        objects with various (optional) attributes.

        Args:
            graphs (list, optional): A list of :class:`deepsnap.graph.Graph`.
            task (str): The task that this :class:`GraphDataset` is used for
                (task = `node` or `edge` or `link_pred` or `graph`).
            custom_split_graphs (list): A list of 2 (train and val)
                or 3 (train, val and test) lists of splitted graphs, used in
                custom split of the `graph` task.
            edge_negative_sampling_ratio (float): The number of negative
                samples compared to that of positive edges. Default value 
                is 1.
            edge_message_ratio (float): The number of message passing edges
                compared to that of training supervision edges. Default value 
                is 0.8.
            edge_train_mode (str): Use `all` or `disjoint`. In `all` 
                mode, training supervision edges are same with the message 
                passing edges. In `disjoint` mode, training supervision 
                objectives are different from the message passing edges. 
                The difference between these two modes please see 
                the `DeepSNAP link prediction Colab <https://colab.research.
                google.com/drive/1ycdlJuse7l2De7wi51lFd_nCuaWgVABc?
                usp=sharing>`_.
            edge_split_mode (str): Use `exact` or `approximate`. This mode is 
                designed for the heterogeneous graph. If the mode is `exact`, 
                split the heterogeneous graph according to both the ratio
                and the split type. If the mode is `approximate`, split the 
                heterogeneous graph regardless of the split type.
            minimum_node_per_graph (int): If the number of nodes of a graph
                is smaller than the minimum node per graph, that graph will 
                be filtered out.
            generator (:class:`Generator`): The dataset will be on-the-fly 
                generated. The on-the-fly generator will be used, if the 
                :obj:`self.graphs` is empty or `None`, and the generator 
                (:class:`Generator`) is provided with an overwritten 
                :meth:`generate` method.
            resample_negatives (bool): Whether to resample negative edges in
                each iteration of the `link_pred` task. User needs to set this
                variable in the case of tensor backend for the custom split.
            resample_disjoint (bool): Whether to resample disjoint training 
                edges in the `disjonint` `link_pred` task.
            resample_disjoint_period (int): The number of iterations after
                which the training edges in the `disjoint` mode are resampled.
            negative_label_val (int, optional): The value of negative edges 
                generated in link_pred task. User needs to set this variable 
                in the case of tensor backend custom split.
            netlib (types.ModuleType, optional): The graph backend module. 
                Currently DeepSNAP supports the NetworkX and SnapX (for 
                SnapX only the undirected homogeneous graph) as the graph 
                backend. Default graph backend is the NetworkX.
        """
    def __init__(
        self,
        graphs: List[Graph] = None,
        task: str = "node",
        custom_split_graphs: List[Graph] = None,
        edge_negative_sampling_ratio: float = 1,
        edge_message_ratio: float = 0.8,
        edge_train_mode: str = "all",
        edge_split_mode: str = "exact",
        minimum_node_per_graph: int = 5,
        generator=None,
        resample_negatives: bool = False,
        resample_disjoint: bool = False,
        resample_disjoint_period: int = 1,
        negative_label_val: int = None,
        netlib=None
    ):
        if netlib is not None:
            deepsnap._netlib = netlib
        if graphs is not None:
            # make sure graphs is a list
            if not isinstance(graphs, list):
                graphs = [graphs]

            # support user' input a list of netlib.Graph instead of Graph
            for i, graph in enumerate(graphs):
                if not isinstance(graph, Graph):
                    graphs[i] = Graph(graph, netlib=netlib)

        # validity check for `task`
        if task not in ["node", "edge", "link_pred", "graph"]:
            raise ValueError(
                "task must be one of node, edge, link_pred or graph."
            )

        # validity check for `edge_train_mode`
        if edge_train_mode not in ["all", "disjoint"]:
            raise ValueError("edge_train_mode must be all or disjoint.")

        # validity check for `edge_split_mode`
        if edge_split_mode not in ["exact", "approximate"]:
            raise ValueError(
                "edge_split_mode must be exact or approximate."
            )

        # validity check for `resample_negatives`
        if resample_negatives and (task != "link_pred"):
            raise ValueError(
                "resample_negatives set to True only make sense "
                "when self.task is link_pred."
            )

        # validity check for `resample_disjoint`
        # if resample_negatives and (self.task != "link_pred"):
        if (
            resample_disjoint
            and (
                (task != "link_pred")
                or (edge_train_mode != "disjoint")
            )
        ):
            raise ValueError(
                "resample_disjoint set to True only make sense "
                "when self.task is `link_pred` and edge_train_mode is "
                "disjoint."
            )
        # validity check for `resample_negatives`
        if (negative_label_val is not None) and (task != "link_pred"):
            raise ValueError(
                "negative_label_val is set only make sense "
                "when self.task is link_pred."
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

        # set private parameters
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
            # by default on-the-fly generator is not used
            # when generator is not provide
            self.generator = None

            # filter graphs that are too small
            if self.minimum_node_per_graph > 0:
                graphs_filter = []
                for idx, graph in enumerate(self.graphs):
                    if isinstance(graph, Graph):
                        if isinstance(graph, HeteroGraph):
                            if (
                                sum(graph.num_nodes().values())
                                >= self.minimum_node_per_graph
                            ):
                                graphs_filter.append(graph)
                            else:
                                warnings.warn(
                                    f"the {idx}-th graph in self.graphs is "
                                    "filtered out as it contains "
                                    f"{sum(graph.num_nodes().values())} nodes,"
                                    " which is less than "
                                    "self.minimum_node_per_graph: "
                                    f"{self.minimum_node_per_graph}."
                                )
                        else:
                            if graph.num_nodes >= self.minimum_node_per_graph:
                                graphs_filter.append(graph)
                            else:
                                warnings.warn(
                                    f"the {idx}-th graph in self.graphs is "
                                    "filtered out as it contains "
                                    f"{graph.num_nodes} nodes,"
                                    " which is less than "
                                    "self.minimum_node_per_graph: "
                                    f"{self.minimum_node_per_graph}."
                                )
                    else:
                        raise TypeError(
                            "element in self.graphs of unexpected type"
                        )
                self.graphs = graphs_filter

            # update graph in self.graphs with appropriate custom
            # split component
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

            # update graph in self.graphs with negative_label_val
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
                                            ).item() + 1
                                        )
                                else:
                                    negative_label_val = max(
                                        negative_label_val,
                                        torch.max(graph.edge_label).item() + 1
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
        r"""
        Create negative edges and labels for tensor backend link_pred case.
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
        Returns the node feature dimension.

        Returns:
            int: The node feature dimension for the graphs 
            in the dataset.
        """
        return self._graph_example.num_node_features

    @property
    def num_node_labels(self) -> int:
        r"""
        Returns the number of node labels.

        Returns:
            int: The number of node labels for the graphs 
            in the dataset.
        """
        if self._num_node_labels is None:
            if self.graphs is None:
                self._num_node_labels = self.generator.num_node_labels
            else:
                unique_node_labels = torch.LongTensor([])
                for graph in self.graphs:
                    unique_node_labels = torch.cat([
                        unique_node_labels, graph.get_num_labels("node_label")
                    ])
                self._num_node_labels = torch.unique(
                    unique_node_labels
                ).shape[0]
        return self._num_node_labels

    @property
    def num_nodes(self) -> List[int]:
        r"""
        Return the number of nodes for the graphs in the dataset.

        Returns:
            list: A list of number of nodes for the graphs 
            in the dataset.
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
        Returns the edge feature dimension.

        Returns:
            int: The edge feature dimension for the graphs 
            in the dataset.
        """
        return self._graph_example.num_edge_features

    @property
    def num_edge_labels(self) -> int:
        r"""
        Returns the number of edge labels.

        Returns:
            int: The number of edge labels for the graphs 
            in the dataset.
        """
        if self._num_edge_labels is None:
            if self.graphs is None:
                self._num_edge_labels = self.generator.num_edge_labels
            else:
                unique_edge_labels = torch.LongTensor([])
                for graph in self.graphs:
                    unique_edge_labels = torch.cat([
                        unique_edge_labels, graph.get_num_labels("edge_label")
                    ])
                self._num_edge_labels = torch.unique(
                    unique_edge_labels
                ).shape[0]
        return self._num_edge_labels

    @property
    def num_edges(self) -> List[int]:
        r"""
        Return the number of edges for the graphs in the dataset.

        Returns:
            list: A list of number of edges for the graphs 
            in the dataset.
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
        Returns the graph feature dimension.

        Returns:
            int: The graph feature dimension for the graphs 
            in the dataset.
        """
        return self._graph_example.num_graph_features

    @property
    def num_graph_labels(self) -> int:
        r"""
        Returns the number of graph labels.

        Returns:
            int: The number of graph labels for the graphs 
            in the dataset.
        """
        if self._num_graph_labels is None:
            if self.graphs is None:
                self._num_graph_labels = self.generator.num_graph_labels
            else:
                unique_graph_labels = torch.LongTensor([])
                for graph in self.graphs:
                    unique_graph_labels = torch.cat([
                        unique_graph_labels,
                        graph.get_num_labels("graph_label")
                    ])
                self._num_graph_labels = torch.unique(
                    unique_graph_labels
                ).shape[0]
        return self._num_graph_labels

    @property
    def num_labels(self) -> int:
        r"""
        A General wrapper that returns the number of labels depending on
        the task.

        Returns:
            int: The number of labels, depending on the task.
        """
        if self.task == "node":
            return self.num_node_labels
        elif self.task == "edge" or self.task == "link_pred":
            return self.num_edge_labels
        elif self.task == "graph":
            return self.num_graph_labels
        else:
            raise ValueError(f"Task {self.task} not supported")

    def num_dims_dict(self) -> Dict[str, int]:
        r"""
        Dimensions of all fields.

        Returns:
            dict: Dimensions of all fields. For example, if 
            graphs have two attributes the `node_feature` 
            and the `graph_label`. The returned dictionary will 
            have two keys, `node_feature` and `graph_label`, and 
            two values, node feature dimension and graph label 
            dimension.
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
        r"""
        Update self.general_splits_mode, self.disjoint_split_mode &
        self.negative_edges_mode to indicate whether we are working on custom
        support for:
            (1) general transductive or inductive custom split
            or (2) disjoint train custom split in disjoint link_pred task
            or (3) custom negative edges in link_pred task
        """
        custom_keys = ["general_splits", "disjoint_split", "negative_edges"]
        for custom_key in custom_keys:
            self[f"{custom_key}_mode"] = "random"

        # make sure custom appeared in all graphs or no custom appeared
        # in any graphs
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
            list: A list of 3 (2) lists of :class:`deepsnap.graph.Graph`
            object corresponding to train, validation (and test) set.
        """
        split_graphs = []
        for graph in self.graphs:
            if self.general_splits_mode == "custom":
                split_graph = graph._custom_split(
                    task=self.task
                )
            elif self.general_splits_mode == "random":
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
                        graph._is_train = True
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
                            graph_temp.negative_edge = (
                                graph_temp.negative_edges[i]
                            )
                            if isinstance(graph_temp, HeteroGraph):
                                graph_temp._custom_create_neg_sampling(
                                    self.edge_negative_sampling_ratio,
                                    split_types=split_types
                                )
                            else:
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
            List[Graph]: a list of 3 (2) lists of graph object corresponding
            to train, validation (and test) set.
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

            # If the `default split` policy would result in empty splited
            # graphs, `secure split` policy would be used instead
            split_empty_flag = False

            split_offset = 0
            # perform `default split`
            for i, split_ratio_i in enumerate(split_ratio):
                if i != len(split_ratio) - 1:
                    num_split_i = int(split_ratio_i * num_graphs)
                    graphs_split_i = (
                        self.graphs[split_offset:split_offset + num_split_i]
                    )
                    split_offset += num_split_i
                else:
                    graphs_split_i = self.graphs[split_offset:]
                if len(graphs_split_i) == 0:
                    split_empty_flag = True
                    split_offset = 0
                    split_graphs = []
                    break
                split_graphs.append(graphs_split_i)

            if split_empty_flag:
                # perform `secure split` s.t. guarantees all splitted graph
                # list contains at least one graph.
                for i, split_ratio_i in enumerate(split_ratio):
                    if i != len(split_ratio) - 1:
                        num_split_i = (
                            1 +
                            int(
                                split_ratio_i
                                * (num_graphs - len(split_ratio))
                            )
                        )
                        graphs_split_i = (
                            self.graphs[
                                split_offset:split_offset + num_split_i
                            ]
                        )
                        split_offset += num_split_i
                    else:
                        graphs_split_i = self.graphs[split_offset:]
                    split_graphs.append(graphs_split_i)

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
                    graph_temp = split_graphs[i][j]
                    if isinstance(graph_temp, Graph):
                        # store the original edge_label
                        graph_edge_label = None
                        if (
                            self.resample_disjoint
                            and (i == 0)
                            and hasattr(graph_temp, "edge_label")
                        ):
                            graph_edge_label = graph_temp.edge_label

                        if isinstance(graph_temp, HeteroGraph):
                            graph_temp = (
                                graph_temp.split_link_pred(
                                    split_types,
                                    self.edge_message_ratio,
                                    self.edge_split_mode
                                )[1]
                            )
                        else:
                            graph_temp = (
                                graph_temp.split_link_pred(
                                    self.edge_message_ratio
                                )[1]
                            )

                        # save the original edge_label
                        if graph_edge_label is not None:
                            graph_temp._edge_label = (
                                copy.deepcopy(graph_edge_label)
                            )
                        else:
                            graph_temp._edge_label = None

                        # set is_train flag
                        if i == 0:
                            graph_temp._is_train = True

                        split_graphs[i][j] = graph_temp
                    else:
                        raise TypeError(
                            "element in self.graphs of unexpected type."
                        )

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
        r"""
        Split the dataset into train, validation (and test) sets.

        Args:
            transductive (bool): Whether the learning is transductive 
                (`True`) or inductive (`False`). Inductive split is 
                always used for the graph-level task, :obj:`self.task` 
                equals to `graph`.
            split_ratio (list): A list of ratios such as
                `[train_ratio, validation_ratio, test_ratio]`.
            split_types (str or list): Types splitted on. Default is `None`.
            shuffle (bool): Whether to shuffle data for the splitting.

        Returns:
            list: A list of 3 (2) :class:`deepsnap.dataset.GraphDataset`
            objects corresponding to the train, validation (and test) sets.
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
        r""" 
        Resample splits of the message passing and supervision edges in the 
        `disjoint` mode.

        .. note::

            If :meth:`apply_transform` (on the message passing graph)
            was used before this resampling, it needs to be
            re-applied after resampling, to update some of the (supervision)
            edges that were in the objectives.
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
        Applies transformation to all graph objects. All graphs in 
        :obj:`self.graphs` will be run by the specified 
        :meth:`transform` function, and then a new 
        :class:`GraphDataset` object will be returned.

        Args:
            transform (callable): User-defined transformation function.
            update_tensor (bool): If the graphs have changed, use the 
                graph to update the stored tensor attributes.
            update_graph (bool): If the tensor attributes have changed, 
                use the attributes to update the graphs.
            deep_copy (bool): If `True`, all graphs will be deepcopied 
                and then fed into the :meth:`transform` function.
                In this case, the :meth:`transform` function also might 
                need to return a `Graph` object.
            **kwargs (optional): Parameters used in the :meth:`transform` function 
                for each `Graph` object.

        Returns:
            :class:`GraphDataset`: A new :class:`GraphDataset` object with 
            transformed graphs.
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
        r""" 
        Filter the graphs in the dataset. Discarding a graph `G` 
        when `filter_fn(G)` is `False`. :meth:`apply_transform` is an 
        analog of the Python `map` function, while :meth:`filter` 
        is an analog of the Python `filter` function.

        Args:
            filter_fn: User-defined filter function that returns `True` 
                (keep) or `False` (discard) the graph object in 
                the dataset.
            deep_copy: If `True`, all graphs will be deepcopied and 
                then fed into the :meth:`filter` function.
            **kwargs: Parameters used in the :meth:`filter` function.

        Returns:
            :class:`GraphDataset`: A new :class:`GraphDataset` object with 
            graphs filtered.
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
        Transfer the graphs in the dataset to specified device.

        Args:
            device (str): Specified device name, such as `cpu` or
                `cuda`.
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
        tensor_backend: bool = False,
        netlib=None
    ) -> List[Graph]:
        r"""
        Transform a :class:`torch_geometric.data.Dataset` object to a 
        list of :class:`deepsnap.grpah.Graph` objects.

        Args:
            dataset (:class:`torch_geometric.data.Dataset`): A 
                :class:`torch_geometric.data.Dataset` object that will be 
                transformed to a list of :class:`deepsnap.grpah.Graph` 
                objects.
            verbose (bool): Whether to print information such as warnings.
            fixed_split (bool): Whether to load the fixed data split from 
                the original PyTorch Geometric dataset.
            tensor_backend (bool): `True` will use pure tensors for graphs.
            netlib (types.ModuleType, optional): The graph backend module. 
                Currently DeepSNAP supports the NetworkX and SnapX (for 
                SnapX only the undirected homogeneous graph) as the graph 
                backend. Default graph backend is the NetworkX.

        Returns:
            list: A list of :class:`deepsnap.graph.Graph` objects.
        """

        if fixed_split:
            graphs = [
                Graph.pyg_to_graph(
                    data, verbose=verbose, fixed_split=True,
                    tensor_backend=tensor_backend, netlib=netlib
                )
                for data in dataset
            ]
            graphs_split = [[graph] for graph in graphs[0]]
            return graphs_split
        else:
            return [
                Graph.pyg_to_graph(
                    data, verbose=verbose,
                    tensor_backend=tensor_backend,
                    netlib=netlib
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
            Union[:class:`deepsnap.graph.Graph`,
            List[:class:`deepsnap.graph.Graph`]]: A single
            :class:`deepsnap.graph.Graph` object or subset
            of :class:`deepsnap.graph.Graph` objects.
        """
        # TODO: add the hetero graph equivalent of these functions
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

        # resample disjoint training data only when the task is
        # disjoint link_pred and self.resample_disjoint is set to True
        if (
            self.task == "link_pred"
            and self.edge_train_mode == "disjoint"
            and self.resample_disjoint
            and graph._is_train
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
