Introduction
============

We talk about the rationale of DeepSNAP, introduce DeepSNAP core modules, and show example implementations.

.. contents::
    :local:
    
Background
----------
We first explain some preliminaries for learning on graphs.

We classify the learning tasks into the following categories, all of which are fully supported by DeepSNAP.
Both classification and regression objectives can be applied to all of the tasks.

* **node**: Node-level tasks makes prediction on labels for nodes. The prediction of each node is made based on node embeddings output by a GNN.
* **edge**: Edge-level tasks makes prediction on labels for edges. The prediction of each edge is made based on a pair of node embeddings corresponding to the endpoints of the edge.
* **link_pred**: Link prediction tasks makes prediction on existance of links (edges). The difference from edge-level tasks, is that it not only needs to make prediction of the edge label, but also have to decide if the edge exists at all. Negative sampling can be used here, so the model learns to predict the non-existence of an edge between 2 nodes. In the simplest version without edge label, the task becomes a binary prediction, where 1 corresponds to existence of an edge and 0 otherwise.
* **graph**: Graph-level tasks makes prediction on labels for graphs. The prediction of each graph is made based on a `pooled` graph embedding from node embeddings. Naive pooling includes simply summing or taking average of all embeddings of nodes in the graph. See PyTorch Geometric for more pooling options.

In the dataset level, for each type of tasks, there are 2 possible types of splits which DeepSNAP fully supports:

* **train / val**: 2 splits, including training set and validation set. E.g., `split_ratio = [0.8, 0.2]`
* **train / val / test**: 3 splits, including training, validation and test set. E.g., `split_ratio = [0.8, 0.1, 0.1]`

Additionally, a type of split can be either transductive or inductive:

* **transductive**: `training`, `validation` and `test` splits include all the graph(s) in the dataset. Within each graph, node or edge labels are split depending on the task.
* **inductive** (only possible with multi-graph datasets): `training`, `validation` and `test` splits include distinct graphs. Within each training graph, all labels observed; within each `validation` / `test` graph, no label is observed.

Moreover, all splits performed in DeepSNAP are `"secure split"`, such that if there are enough splitted objects, all splitted graphs are guaranteed with at least one object. (Case with not enough objects refers to graph with less splitted objects than the number of splits. E.g. there are 2 nodes in the graph, while the user want to split nodes in the graph to `train` / `val` / `test`.)
E.g. Consider a graph with 5 edges and we would like to split the graph to `train` / `val` / `test` with `split_ratio = [0.8, 0.1, 0.1]`, then without `"secure split"`, the number of edges in each splitted graph will be 4, 0, 1, resulting in one splitted graph with 0 splitted objects. With the `"secure split"`, we will preprocess the splitted objects by having 2 or 3 (depending on whether we would split the graph to 2 or 3 parts) held out objects, and apply the same splitting logic on the remaining splitted objects, which in our case would have the number of edges in splitted graph results in 2, 1, 2.

* **minimum_node_per_graph**: filtering out graphs with not enough splitted objects, all graphs imported in :class:`deepsnap.dataset.GraphDataset` with number of nodes less than `minimum_node_per_graph` will be automatically removed. If `minimum_node_per_graph` is not specified by user, it will have a default value of 5.

DeepSNAP Graph
--------------

The :class:`deepsnap.graph.Graph` class is responsible for manipulating a graph object for training GNNs. The most important functionalities of Graph object include

* Splitting a graph into `train`, `validation`, `test` (in the `transductive` setting) and performing negative sampling for link prediction task.
* Applying a user-defined transform function, and ensures that the graph backend is in sync with the tensor representation of graphs used for GNNs.

The first way to create a DeepSNAP :class:`deepsnap.graph.Graph` is to load 
from a NetworkX graph object. The following is an example to create a complete 
graph by using the NetworkX.

.. code-block:: python
	
    import networkx as nx
    from deepsnap.graph import Graph

    G = nx.complete_graph(100)
    H = Graph(G)
    >>> Graph(G=[], edge_index=[2, 9900], edge_label_index=[2, 9900], node_label_index=[100])

User can also create a :class:`deepsnap.graph.Graph` from the PyTorch Geometric data format directly.

.. code-block:: python

    from deepsnap.graph import Graph
    from torch_geometric.datasets import Planetoid

    pyg_dataset = Planetoid('./cora', 'Cora')
    graph = Graph.pyg_to_graph(pyg_dataset[0])
    >>> Graph(G=[], edge_index=[2, 10556], edge_label_index=[2, 10556], node_feature=[2708, 1433], node_label=[2708], node_label_index=[2708])

When creating a DeepSNAP graph, any NetworkX attribute begin with :attr:`node_`, :attr:`edge_`, :attr:`graph_` will be automatically loaded.
When loading from PyTorch Geometric, we automatically renaming the attributes to our naming taxonomy.
Important attributes are listed below:

- :attr:`Graph.node_feature`: Node features.
- :attr:`Graph.node_label`: Node labels.
- :attr:`Graph.edge_feature`: Edge features.
- :attr:`Graph.edge_label`: Edge labels.
- :attr:`Graph.graph_feature`: Graph features.
- :attr:`Graph.graph_label`: Graph labels.

After loading these features, DeepSNAP Graph creates :attr:`index` that are necessary for GNN computation or indicating dataset split.
Important indices are listed below:

- :attr:`Graph.edge_index`: Edge index that guides GNN message passing
- :attr:`Graph.node_label_index`: Slicing node label to get the corresponding split :attr:`G.node_label[G.node_label_index]`.
- :attr:`Graph.edge_label_index`: Slicing edge label to get the corresponding split :attr:`G.edge_label[G.edge_label_index]`.


Following is an example to create a DeepSNAP graph object with node features, we can store the node features in the NetworkX graph with
attribute name :attr:`node_feature`.

.. code-block:: python
    
    import torch
    import networkx as nx
    rom deepsnap.graph import Graph

    G = nx.Graph()
    G.add_node(0, node_feature=torch.tensor([1,2,3]))
    G.add_node(1, node_feature=torch.tensor([4,5,6]))
    G.add_edge(0, 1)
    H = Graph(G)
    print(H.node_feature)
    >>> tensor([[1, 2, 3],
                [4, 5, 6]])

Here is another example to transform a DeepSNAP graph by adding clustering coefficient into the graph object:

.. code-block:: python

    import networkx as nx
    from deepsnap.graph import Graph
    from torch_geometric.datasets import Planetoid

    def clustering_func(graph):
        clustering = list(nx.clustering(graph.G).values())
        graph['node_clustering'] = clustering

    pyg_dataset = Planetoid('./cora', 'Cora')
    graph = Graph.pyg_to_graph(pyg_dataset[0])
    graph.apply_transform(clustering_func, update_graph=True, update_tensor=False)
    print(graph)
    print(graph.G.nodes(data=True)[0])
    >>> Graph(G=[], edge_index=[2, 10556], edge_label_index=[2, 10556], node_clustering=[2708], node_feature=[2708, 1433], node_label=[2708], node_label_index=[2708])
    >>> {'node_feature': tensor([0., 0., 0.,  ..., 0., 0., 0.]), 'node_label': tensor(3), 'node_clustering': 0.3333333333333333}


DeepSNAP Dataset
----------------

The :class:`deepsnap.dataset.GraphDataset` class holds and manipulates a set of DeepSNAP graphs used for `training`, `validation` and / or `testing`. The most important functionalities of the :class:`GraphDataset` object include

* Load standard fixed splits, if available.
* Random `transductive` and `inductive` splitting of a dataset into `training`, `validation` and `test` DeepSNAP Datasets. 
* Applying a user-defined transform function, and ensures that the graph backend is in sync with the tensor representation of graphs used for GNNs.

Dataset splitting encompasses following design choices:

* `inductive` vs `transductive`: The `inductive` setting (for dataset with multiple graphs) splits the dataset by graphs. Distinct sets of graphs are used for `training`, `validation` and `test`, and the test graphs are never seen during training. This can be done for node, edge and graph-level tasks. In the `transductive` setting, all graphs are seen during training time, but the labels for certain nodes and edges are not observed at training time, and are used for validation and test. This applies to node and edge-level tasks.
* Negative sampling is availabe for link prediction by using DeepSNAP, since this is typically an imbalanced tasks due to sparsity of graphs. DeepSNAP provides the option for user to specify the ratio of positive links and negative links for training, validation and test, as well as when to resample negative links during training.
* Disjoint objective (supervision) sampling for link prediction is an important technique often not mentioned in research papers. At training time, it further splits the training set into edges used for message passing, and edges used for link prediction objectives. The rationale is to allow the model to learn to predict unseen edges, instead of memorizing all training edges at training time and failing to generalize to unseen edges at validation and test time. DeepSNAP also supports disjoint objectives and resampling of the disjoint objectives at training time. 

It is convenient to create a DeepSNAP dataset from a list of DeepSNAP graphs.

.. code-block:: python

    import networkx as nx
    from deepsnap.graph import Graph
    from deepsnap.dataset import GraphDataset

    G = nx.complete_graph(100)
    H1 = Graph(G)
    H2 = H1.clone()
    dataset = GraphDataset(graphs=[H1, H2])
    len(dataset)
    >>> 2

DeepSNAP also supports creating the dataset from the PyTorch Geometric datasets directly.

.. code-block:: python

    from deepsnap.dataset import GraphDataset
    from torch_geometric.datasets import TUDataset

    pyg_dataset = TUDataset('./enzymes', 'ENZYMES')
    graphs = GraphDataset.pyg_to_graphs(pyg_dataset)
    dataset = GraphDataset(graphs, task="graph", minimum_node_per_graph=0)
    print(dataset)
    >>> GraphDataset(600)

With the :class:`deepsnap.dataset.GraphDataset`, user can specify the related tasks and DeepSNAP will 
perform functions according to the speficied task.
The tasks include:

- **node**: Node classification.
- **edge**: Edge classification.
- **link_pred**: Link prediction.
- **graph**: Graph classification.

Following is an example to perform a split to `train`, `validation` and `test` set with respect to the `node` 
(node classification) task.

.. code-block:: python

    import torch
    import networkx as nx
    from deepsnap.graph import Graph
    from deepsnap.dataset import GraphDataset

    G = nx.complete_graph(100)
    Graph.add_node_attr(G, 'node_feature', torch.zeros([100, 1]))
    Graph.add_node_attr(G, 'node_label', torch.zeros([100, 1]))
    H1 = Graph(G)
    H2 = H1.clone()
    dataset = GraphDataset(graphs=[H1, H2], task='node')

    train, val, test = dataset.split(transductive=True, split_ratio=[0.8, 0.1, 0.1])
    print(train, val, test)
    >>> GraphDataset(2) GraphDataset(2) GraphDataset(2)

Notice user can also specify whether the learning is `transductive`. In the example above, the nodes in each 
graph is splited to `train`, `validation` and `test` sets with repsect to the `split_ratio` 8:1:1.
If the `transductive` is `False`, the dataset will be splitted as following:

.. code-block:: python

    from deepsnap.dataset import GraphDataset
    from torch_geometric.datasets import TUDataset

    pyg_dataset = TUDataset('./enzymes', 'ENZYMES')
    graphs = GraphDataset.pyg_to_graphs(pyg_dataset)
    dataset = GraphDataset(graphs, task="graph", minimum_node_per_graph=0)
    train, val, test = dataset.split(
                transductive=False, split_ratio = [0.8, 0.1, 0.1])
    print(train, val, test)
    >>> GraphDataset(480) GraphDataset(60) GraphDataset(60)

It is also possible to transform the dataset directly. Here is an example for transforming a DeepSNAP dataset:

.. code-block:: python

    import networkx as nx
    from deepsnap.dataset import GraphDataset
    from torch_geometric.datasets import TUDataset

    def clustering_func(graph):
        clustering = list(nx.clustering(graph.G).values())
        graph['node_clustering'] = clustering

    pyg_dataset = TUDataset('./enzymes', 'ENZYMES')
    graphs = GraphDataset.pyg_to_graphs(pyg_dataset)
    dataset = GraphDataset(graphs, task='graph', minimum_node_per_graph=0)
    dataset.apply_transform(clustering_func, update_graph=True, update_tensor=False)
    print(dataset)
    print(dataset[0])
    >>> GraphDataset(600)
    >>> Graph(G=[], edge_index=[2, 168], edge_label_index=[2, 168], graph_label=[1], node_clustering=[37], node_feature=[37, 3], node_label_index=[37])


DeepSNAP Batch
--------------

The main purpose of the :class:`deepsnap.batch.Batch` is to :meth:`collate` the dataset and make it to be easily used 
with the :class:`torch.utils.data.DataLoader`.
The following example is to :meth:`collate` the train dataset into batches with 10 graphs in each batch.

.. code-block:: python

    import networkx as nx
    from deepsnap.batch import Batch
    from deepsnap.dataset import GraphDataset
    from torch_geometric.datasets import TUDataset
    from torch.utils.data import DataLoader

    def clustering_func(graph):
        clustering = list(nx.clustering(graph.G).values())
        graph['node_clustering'] = clustering

    pyg_dataset = TUDataset('./enzymes', 'ENZYMES')
    graphs = GraphDataset.pyg_to_graphs(pyg_dataset)
    dataset = GraphDataset(graphs, task='graph', minimum_node_per_graph=0)
    train, val, test = dataset.split(
                transductive=False, split_ratio = [0.8, 0.1, 0.1])
    train_loader = DataLoader(train, collate_fn=Batch.collate(), batch_size=10, shuffle=True)
    batch = next(iter(train_loader))
    batch = batch.apply_transform(clustering_func, update_graph=True, update_tensor=False)
    print(batch)
    >>> Batch(G=[10], batch=[266], edge_index=[2, 1064], edge_label_index=[2, 1064], graph_label=[10], node_clustering=[10], node_feature=[266, 3], node_label_index=[266])


Here is another example to transform a DeepSNAP Batch by adding the clustering coefficient to the :attr:`node_feature`:

.. code-block:: python

    import torch
    import networkx as nx
    from deepsnap.batch import Batch
    from deepsnap.dataset import GraphDataset
    from torch_geometric.datasets import TUDataset
    from torch.utils.data import DataLoader

    def clustering_func(graph):
        clustering = torch.tensor(list(nx.clustering(graph.G).values()))
        clustering = clustering.view(-1, 1)
        graph.node_feature = torch.cat([graph.node_feature, clustering], dim=1)

    pyg_dataset = TUDataset('./enzymes', 'ENZYMES')
    graphs = GraphDataset.pyg_to_graphs(pyg_dataset)
    dataset = GraphDataset(graphs, task='graph', minimum_node_per_graph=0)
    train, val, test = dataset.split(
                transductive=False, split_ratio = [0.8, 0.1, 0.1])
    train_loader = DataLoader(train, collate_fn=Batch.collate(), batch_size=10, shuffle=True)
    batch = next(iter(train_loader))
    batch = batch.apply_transform(clustering_func, update_graph=True, update_tensor=False)
    print(batch)
    >>> Batch(G=[10], batch=[411], edge_index=[2, 1378], edge_label_index=[2, 1378], graph_label=[10], node_feature=[411, 4], node_label_index=[411])
    print(nx.get_node_attributes(batch.G[0], 'node_feature')[0].shape[0])
    >>> 4


To have a better understanding of using DeepSNAP with homogeneous graphs, we recommend you to look at the examples:

- `Node classification <https://github.com/snap-stanford/deepsnap/tree/master/examples/node_classification>`__
- `Link prediction <https://github.com/snap-stanford/deepsnap/tree/master/examples/link_prediction>`__
- `Graph classification <https://github.com/snap-stanford/deepsnap/tree/master/examples/graph_classification>`__

Or see our `Colab Notebooks <colab.html>`_.

DeepSNAP Heterogeneous Graph
----------------------------

The DeepSNAP provides :class:`deepsnap.hetero_graph.HeteroGraph` class for the heterogeneous graph.
The main idea is similar to the DeepSNAP :class:`Graph` class. But :class:`deepsnap.hetero_graph.HeteroGraph` 
add some extra peroperties for heterogeneous graph and functions in the class are overrided for the 
heterogeneous graph.

The first way to create a DeepSNAP :class:`deepsnap.hetero_graph.HeteroGraph` is to load
from a NetworkX graph object. The following is an example to create a simple
:class:`HeteroGraph` object by using the NetworkX.

.. code-block:: python

    import torch
    import networkx as nx
    from deepsnap.hetero_graph import HeteroGraph

    G = nx.DiGraph()
    G.add_node(0, node_type='n1', node_label=1, node_feature=torch.Tensor([0.1, 0.2, 0.3]))
    G.add_node(1, node_type='n1', node_label=0, node_feature=torch.Tensor([0.2, 0.3, 0.4]))
    G.add_node(2, node_type='n2', node_label=1, node_feature=torch.Tensor([0.3, 0.4, 0.5]))
    G.add_edge(0, 1, edge_type='e1')
    G.add_edge(0, 2, edge_type='e1')
    G.add_edge(1, 2, edge_type='e2')
    H = HeteroGraph(G)
    for hetero_feature in H:
        print(hetero_feature)

    >>> ('G', <networkx.classes.digraph.DiGraph object at 0x103642370>)
        ('edge_index', {('n1', 'e1', 'n1'): tensor([[0],
                [1]]), ('n1', 'e1', 'n2'): tensor([[0],
                [0]]), ('n1', 'e2', 'n2'): tensor([[1],
                [0]])})
        ('edge_label_index', {('n1', 'e1', 'n1'): tensor([[0],
                [1]]), ('n1', 'e1', 'n2'): tensor([[0],
                [0]]), ('n1', 'e2', 'n2'): tensor([[1],
                [0]])})
        ('edge_to_graph_mapping', {('n1', 'e1', 'n1'): tensor([0]), ('n1', 'e1', 'n2'): tensor([1]), ('n1', 'e2', 'n2'): tensor([2])})
        ('edge_to_tensor_mapping', tensor([0, 0, 0]))
        ('edge_type', {('n1', 'e1', 'n1'): ['e1'], ('n1', 'e1', 'n2'): ['e1'], ('n1', 'e2', 'n2'): ['e2']})
        ('node_feature', {'n1': tensor([[0.1000, 0.2000, 0.3000],
                [0.2000, 0.3000, 0.4000]]), 'n2': tensor([[0.3000, 0.4000, 0.5000]])})
        ('node_label', {'n1': tensor([1, 0]), 'n2': tensor([1])})
        ('node_label_index', {'n1': tensor([0, 1]), 'n2': tensor([0])})
        ('node_to_graph_mapping', {'n1': tensor([0, 1]), 'n2': tensor([2])})
        ('node_to_tensor_mapping', tensor([0, 1, 0]))
        ('node_type', {'n1': ['n1', 'n1'], 'n2': ['n2']})

User can also create a :class:`deepsnap.hetero_graph.HeteroGraph` from the PyTorch Geometric data format directly
in similar manner of the homogeneous graph case.

When creating a DeepSNAP heterogeneous graph, any NetworkX attribute begin with :attr:`node_`, :attr:`edge_`, :attr:`graph_` will be automatically loaded.
Important attributes are listed below:

- :attr:`HeteroGraph.node_feature`: Node features.
- :attr:`HeteroGraph.node_label`: Node labels.
- :attr:`HeteroGraph.edge_feature`: Edge features.
- :attr:`HeteroGraph.edge_label`: Edge labels.
- :attr:`HeteroGraph.graph_feature`: Graph features.
- :attr:`HeteroGraph.graph_label`: Graph labels.

After loading these features, DeepSNAP Graph creates :attr:`index` that are necessary for GNN computation or indicating dataset split.
Important indices are listed below:

- :attr:`HeteroGraph.edge_index`: Edge index that guides GNN message passing
- :attr:`HeteroGraph.node_label_index`: Slicing node label to get the corresponding split :attr:`G.node_label[G.node_label_index]`.
- :attr:`HeteroGraph.edge_label_index`: Slicing edge label to get the corresponding split :attr:`G.edge_label[G.edge_label_index]`.

Similar to the homogeneous graph, the :class:`HeteroGraph` also includes a NetworkX backend graph object for applying transform functions.
Note that the node type for each node has to be specified as a node property :attr:`node_type` in the NetworkX graph object. Similarly, the edge type for each edge has to be specified as an edge property :attr:`edge_type` in the NetworkX graph object. 
The :class:`deepsnap.hetero_graph.HeteroGraph` will store the some data in a `dict` format.
For example, :attr:`HeteroGraph.node_feature` is a dictionary of :attr:`node_type` as keys and values are the node 
feature tensors for each :attr:`node_type`. Similarly, :attr:`HeteroGraph.edge_feature` is a dictionary of :attr:`edge_type` as keys and
values are the edge features for each :attr:`edge_type`.

The heterogeneous GNN framework is fully general and supports both heterogeneity of nodes and edges. It defines the concept of
:attr:`message_types`, as `tuples` in the format of `(start_node_type, edge_type, end_node_type)`. A single node / edge type is used if there is only 1 type of node or edges. The messages for different message types can be parameterized by different weights or even different message passing model.
For example, :attr:`HeteroGraph.edge_index` and :attr:`HeteroGraph.edge_label_index` are dictionaries of :attr:`message_types`
as keys and values are :class:`torch.Tensor` representing edge indices of each :attr:`message_type`.

Dataset splitting for heterogeneous graph encompasses the following additional design choices:

* :attr:`split_types` is a heterogeneous graph specific parameter to let the user specify which types the user would like to split in
  the splitting process for the user specified :attr:`task`. To be more specific, for node split task, the :attr:`split_types` could be either
  a :attr:`node_type` or a list of :attr:`node_type`, and for edge split task and link prediction task, the :attr:`split_types` could be either
  a :attr:`message_type` or a list of :attr:`message_type`. Note that if :attr:`split_types` is not specified in the split function, then the
  default behavior is to include all types corresponding to the :attr:`task`.

* :attr:`edge_split_mode` is a heterogeneous graph specific parameter to let the user specify whether to use some extra resources to have
  edges of each :attr:`message_type` respect to the :attr:`split_ratio` as well.
  :attr:`edge_split_mode` could either be set to `exact` or `approximate`. If `exact` is set, and when :attr:`task` is set to link prediction
  task, then in the splitting process, the relative number of edges for each :attr:`message_type` is exactly splitted correspnding to
  the :attr:`split_ratio`. If `approximate` is set, and when :attr:`task` is set to link prediction task, then in the splitting process,
  even though the total number of edges will be exactly splitted corresponding to the :attr:`split_ratio`, this relative split ratio
  might not hold for edges within each :attr:`message_type`. Note that if :attr:`edge_split_mode` is not specified in the initilization process,
  then the default behavior is `exact`. Additionally, when the :attr:`split_types` includes all types
  of object in its corresponding :attr:`task`, having :attr:`edge_split_mode` set to `approximate` could give the user some performance gain.


DeepSNAP Heterogeneous GNN
--------------------------

The Heterogeneous GNN layer is a PyTorch :class:`nn.Module` that supports easy creation of heterogeneous GNN, building on top of PyTorch Geometric. Users can easily specify the message passing model for each message type.
The message passing models are straightforward adaptation of Pytorch Geometric homogeneous models (such as GraphSAGE, GCN, GIN). In future release, we will provide even easier utilities to create such heterogeneous message passing models.

An example GNN layer for heterogeneous graph is :class:`deepsnap.hetero_gnn.HeteroSAGEConv`.

The module :class:`deepsnap.hetero_gnn.HeteroConv` allows heterogeneous message passing for all message types to be performed on a
heterogeneous graph, which acts like a `wrapper` layer.

There are also some useful functions for the heterogeneous GNN, such as the :func:`deepsnap.hetero_gnn.forward_op` 
and :func:`deepsnap.hetero_gnn.loss_op`, which are helpful to build the heterogeneous GNN model.


For more details on :class:`deepsnap.hetero_graph.HeteroGraph`, please see DeepSNAP examples for heterogeneous graph:

- `Node classification <https://github.com/snap-stanford/deepsnap/tree/master/examples/node_classification_hetero>`__
- `Link prediction <https://github.com/snap-stanford/deepsnap/tree/master/examples/link_prediction_hetero>`__

Or see our `Colab Notebooks <colab.html>`_.