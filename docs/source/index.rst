:github_url: https://github.com/snap-stanford/deepsnap

DeepSNAP Documentation
======================

DeepSNAP is a Python library to assist efficient deep learning on graphs. DeepSNAP features in its support for flexible graph manipulation, standard pipeline, heterogeneous graphs and simple API.

Graph neural networks (GNNs) have achieved state-of-the-art in many domains (to name a few, chemistry, biology, physics, social networks, knowledge graphs), and several efficient implementation frameworks have been proposed, including `PyTorch Geometric <https://pytorch-geometric.readthedocs.io/en/latest/>`_, `DGL <https://www.dgl.ai/>`_, `GraphNets <https://github.com/deepmind/graph_nets>`_ among many others.
However, when developing new GNNs, or applying GNNs to domain-specific areas, several important challenges remain for users:

* Sophisticated graph manipulations are needed during training, such as feature computation, pretraining, subgraph extraction etc., sometimes even needed for every training iteration.
* In most frameworks, standard pipelines for node, edge, link, graph-level tasks under inductive or transductive settings are left to the user to code. In practice, there are additional design choices involved (such as how to split dataset for link prediction). Having a standard pipeline greatly saves repetitive coding efforts, and enables fair comparision for models.
* Many real-world graphs are heterogeneous. General support for heterogeneous graphs, including data storage and flexible message passing, is lacking.

DeepSNAP bridges powerful graph libraries such as `NetworkX <https://networkx.github.io/>`_ and deep learning framework `PyTorch Geometric <https://pytorch-geometric.readthedocs.io/en/latest/>`_. With an intuitive and easy-than-ever API, DeepSNAP addresses the above pain points:

* DeepSNAP currently supports a Networkx-based backend (also SnapX-based backend for homogeneous undirected graph), allowing users to seamlessly call hundreds of graph algorithms available to manipulate / transform the graphs, even at every training iteration.
* DeepSNAP provides a standard pipeline for dataset split, negative sampling and defining node/edge/graph-level objectives, which are transparent to users.
* DeepSNAP provides efficient support for flexible and general heterogeneous GNNs, that supports both node and edge heterogeneity, and allows users to control how messages are parameterized and passed.
* DeepSNAP has an easy-to-use API that works seamlessly with existing GNN models / datasets implemented in PyTorch Geometric. There is close to zero learning curve if the user is familiar with PyTorch Geometric.


.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Overview

   notes/installation
   notes/introduction
   notes/colab
   notes/other

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Package Reference

   modules/deepsnap
   modules/batch
   modules/dataset
   modules/graph
   modules/hetero_gnn
   modules/hetero_graph

.. Indices and Tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
