import torch
from deepsnap.graph import Graph
from deepsnap.hetero_graph import HeteroGraph
from typing import (
    Callable,
    Dict,
    List
)


class Batch(Graph):
    r"""
    A plain old python object modeling a batch of
    :class:`deepsnap.graph.Graph` objects as one big (disconnected) graph,
    with :class:`torch_geometric.data.Data` being the
    base class that all its methods can also be used here.
    In addition, graphs can be reconstructed via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier.

    .. note::

        For more detailed use of :class:`deepsnap.batch.Batch`, see the `examples 
        <https://github.com/snap-stanford/deepsnap/tree/master/examples>`_ folder.

    """
    def __init__(self, batch=None, **kwargs):
        super(Batch, self).__init__(**kwargs)

        self.batch = batch
        self.__data_class__ = Graph
        self.__slices__ = None

    @staticmethod
    def collate(follow_batch=[], transform=None, **kwargs):
        return lambda batch: Batch.from_data_list(
            batch, follow_batch, transform, **kwargs
        )

    @staticmethod
    def from_data_list(
        data_list: List[Graph],
        follow_batch: List = None,
        transform: Callable = None,
        **kwargs
    ):
        r"""
        Constructs A :class:`deepsnap.batch.Batch` object from a python list
        holding :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly.
        Additionally, creates assignment batch vectors for each key in
        :obj:`follow_batch`.

        Args:
            data_list (list): A list of :class:`deepsnap.graph.Graph` objects.
            follow_batch (list): Creates assignment batch vectors
                for each key.
            transform (callable): If it is not `None`, apply transform 
                when batching.
            **kwargs: Other parameters.
        """
        if follow_batch is None:
            follow_batch = []
        if transform is not None:
            data_list = [
                data.apply_transform(
                    transform,
                    deep_copy=True,
                    **kwargs,
                )
                for data in data_list
            ]
        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))
        assert "batch" not in keys

        batch, cumsum = Batch._init_batch_fields(keys, follow_batch)
        batch.__data_class__ = data_list[0].__class__
        batch.batch = []
        for i, data in enumerate(data_list):
            # Note: in heterogeneous graph, __inc__ logic is different
            Batch._collate_dict(
                data, cumsum,
                batch.__slices__, batch,
                data, follow_batch, i=i
            )
            if isinstance(data, Graph):
                if isinstance(data, HeteroGraph):
                    num_nodes = sum(data.num_nodes().values())
                else:
                    num_nodes = data.num_nodes
            else:
                raise TypeError(
                    "element in self.graphs of unexpected type"
                )
            if num_nodes is not None:
                item = torch.full((num_nodes, ), i, dtype=torch.long)
                batch.batch.append(item)

        if num_nodes is None:
            batch.batch = None

        Batch._dict_list_to_tensor(batch, data_list[0])

        return batch.contiguous()

    @staticmethod
    def _init_batch_fields(keys, follow_batch):
        batch = Batch()
        batch.__slices__ = {key: [0] for key in keys}

        for key in keys:
            batch[key] = []

        for key in follow_batch:
            batch[f"{key}_batch"] = []

        cumsum = {key: 0 for key in keys}
        return batch, cumsum

    @staticmethod
    def _collate_dict(
        curr_dict,
        cumsum: Dict[str, int],
        slices,
        batched_dict,
        graph,
        follow_batch,
        i=None
    ):
        r""" Called in from_data_list to collate a dictionary.
        This can also be applied to Graph object, since it has support for
        keys and __getitem__().

        Args:
            curr_dict: current dictionary to be added to the
                collated dictionary.
            cumsum: cumulative sum to be used for indexing.
            slices: a dictionary of the same structure as batched_dict,
                slices[key] indicates the indices to slice batch[key] into
                tensors for all graphs in the batch.
            batched_dict: the batched dictionary of the same structure
                as curr_dict. But all graph data are batched together.
        """
        if isinstance(curr_dict, dict):
            keys = curr_dict.keys()
        else:
            keys = curr_dict.keys
        for key in keys:
            item = curr_dict[key]
            if isinstance(item, dict):
                # recursively collate every key in the dictionary
                if isinstance(batched_dict[key], list):
                    # nested dictionary not initialized yet
                    assert len(batched_dict[key]) == 0
                    # initialize the nested dictionary for batch
                    cumsum[key] = {inner_key: 0 for inner_key in item.keys()}
                    slices[key] = {inner_key: [0] for inner_key in item.keys()}
                    batched_dict[key] = {}
                    for inner_key in item.keys():
                        batched_dict[key][inner_key] = []
                    for inner_key in follow_batch:
                        batched_dict[key][f"{key}_batch"] = []
                Batch._collate_dict(
                    item, cumsum[key],
                    slices[key], batched_dict[key],
                    graph, follow_batch, i=i
                )
                continue
            if torch.is_tensor(item) and item.dtype != torch.bool:
                item = item + cumsum[key]
            if torch.is_tensor(item):
                size = item.size(graph.__cat_dim__(key, curr_dict[key]))
            else:
                size = 1
            slices[key].append(size + slices[key][-1])
            cumsum[key] = cumsum[key] + graph.__inc__(key, item)
            batched_dict[key].append(item)

            if key in follow_batch:
                item = torch.full((size, ), i, dtype=torch.long)
                batched_dict[f"{key}_batch"].append(item)

    @staticmethod
    def _dict_list_to_tensor(dict_of_list, graph):
        r"""Convert a dict/Graph with list as values to a dict/Graph with
        concatenated/stacked tensor as values.
        """
        if isinstance(dict_of_list, dict):
            keys = dict_of_list.keys()
        else:
            keys = dict_of_list.keys
        for key in keys:
            if isinstance(dict_of_list[key], dict):
                # recursively convert the dictionary of list to dict of tensor
                Batch._dict_list_to_tensor(dict_of_list[key], graph)
                continue
            item = dict_of_list[key][0]
            if torch.is_tensor(item):
                if (
                    Graph._is_graph_attribute(key)
                    and item.ndim == 1
                    and (not item.dtype == torch.long)
                    and "feature" in key
                ):
                    # special consideration: 1D tensor for graph
                    # attribute (classification)
                    # named as: "graph_xx_feature"
                    # batch by stacking the first dim
                    dict_of_list[key] = torch.stack(
                        dict_of_list[key],
                        dim=0
                    )
                else:
                    # concat at the __cat_dim__
                    dict_of_list[key] = torch.cat(
                        dict_of_list[key],
                        dim=graph.__cat_dim__(key, item)
                    )
            elif isinstance(item, (float, int)):
                dict_of_list[key] = torch.tensor(dict_of_list[key])

    def to_data_list(self):
        r"""
        Reconstructs the list of :class:`torch_geometric.data.Data` objects
        from the batch object.
        The batch object must have been created via :meth:`from_data_list` in
        order to be able to reconstruct the initial objects.
        """
        if self.__slices__ is None:
            raise RuntimeError(
                "Cannot reconstruct data list from batch because the "
                "batch object was not created using Batch.from_data_list()"
            )

        keys = [key for key in self.keys if key[-5:] != "batch"]
        cumsum = {key: 0 for key in keys}
        data_list = []
        for i in range(len(self.__slices__[keys[0]]) - 1):
            # i: from 0 up to num graphs in the batch
            data = self.__data_class__()
            self._reconstruct_dict(
                i, keys, data, cumsum, self.__slices__, self, data
            )
            data_list.append(data)

        return data_list

    def _reconstruct_dict(
            self, graph_idx: int, keys, data_dict,
            cumsum: Dict[str, int], slices, batched_dict, graph):

        for key in keys:
            if isinstance(batched_dict[key], dict):
                # recursively unbatch the dict
                data_dict[key] = {}
                inner_keys = [
                    inner_key
                    for inner_key in batched_dict[key].keys()
                    if inner_key[-5:] != "batch"
                ]
                inner_cumsum = {inner_key: 0 for inner_key in inner_keys}
                inner_slices = slices[key]
                self._reconstruct_dict(
                    graph_idx, inner_keys,
                    data_dict[key], inner_cumsum,
                    inner_slices, batched_dict[key], graph
                )
                continue

            if torch.is_tensor(batched_dict[key]):
                data_dict[key] = batched_dict[key].narrow(
                    graph.__cat_dim__(key, batched_dict[key]),
                    slices[key][graph_idx],
                    slices[key][graph_idx + 1] - slices[key][graph_idx]
                )
                if batched_dict[key].dtype != torch.bool:
                    data_dict[key] = data_dict[key] - cumsum[key]
            else:
                data_dict[key] = (
                    batched_dict[key][
                        slices[key][graph_idx]:slices[key][graph_idx + 1]
                    ]
                )
            cumsum[key] = cumsum[key] + graph.__inc__(key, data_dict[key])

    @property
    def num_graphs(self) -> int:
        r"""
        Returns the number of graphs in the batch.

        Returns:
            int: The number of graphs in the batch.
        """
        return self.batch[-1].item() + 1

    def apply_transform(
        self,
        transform,
        update_tensor: bool = True,
        update_graph: bool = False,
        deep_copy: bool = False,
        **kwargs
    ):
        r"""
        Applies a transformation to each graph object in parallel by first
        calling `to_data_list`, applying the transform, and then perform
        re-batching again to a `Batch`.
        A transform should edit the graph object,
        including changing the graph structure, or adding
        node / edge / graph level attributes.
        The rest are automatically handled by the
        :class:`deepsnap.graph.Graph` object, including everything
        ended with `index`.

        Args:
            transform (callable): Transformation function applied to each graph object.
            update_tensor (bool): Whether use nx graph to update tensor attributes.
            update_graph (bool): Whether use tensor attributes to update nx graphs.
            deep_copy (bool): :obj:`True` if a new deep copy of batch is returned.
                This option allows modifying the batch of graphs without
                changing the graphs in the original dataset.
            kwargs: Parameters used in the transform function for each
                :class:`deepsnap.graph.Graph`.

        Returns:
            A batch object containing all transformed graph objects.

        """
        # TODO: transductive setting, assert update_tensor == True
        return self.from_data_list(
            [
                Graph(graph).apply_transform(
                    transform, update_tensor, update_graph, deep_copy, **kwargs
                )
                for graph in self.G
            ]
        )

    def apply_transform_multi(
        self,
        transform,
        update_tensors: bool = True,
        update_graphs: bool = False,
        deep_copy: bool = False,
        **kwargs
    ):
        r"""
        Compared to :meth:`apply_transform`, this allows multiple graph objects
        to be returned by the given transform function.

        Args:
            transform (callable): (Multiple return value) tranformation function
                applied to each graph object. It needs to return a tuple of
                Graph objects.
            update_tensors (bool): Whether use nx graph to update tensor attributes.
            update_graphs (bool): Whether use tensor attributes to update nx graphs.
            deep_copy (bool): :obj:`True` if a new deep copy of batch is returned.
                This option allows modifying the batch of graphs without
                changing the graphs in the original dataset.
            kwargs: Parameters used in the transform function for each
                :class:`deepsnap.graph.Graph`.

        Returns:
            A tuple of batch objects. The i-th batch object contains the i-th
            return value of the transform function applied to all graphs
            in the batch.
        """
        g_lists = (
            zip(
                *[
                    Graph(graph).apply_transform_multi(
                        transform, update_tensors, update_graphs,
                        deep_copy, **kwargs,
                    )
                    for graph in self.G
                ]
            )
        )
        return (self.from_data_list(g_list) for g_list in g_lists)
