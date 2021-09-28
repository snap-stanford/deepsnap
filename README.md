# DeepSNAP

[![PyPI](https://img.shields.io/pypi/v/deepsnap.svg?color=brightgreen)](https://pypi.org/project/deepsnap/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/snap-stanford/deepsnap/blob/master/LICENSE)
[![Build Status](https://github.com/snap-stanford/deepsnap/actions/workflows/test.yml/badge.svg)](https://github.com/snap-stanford/deepsnap/actions/workflows/test.yml)
[![Code Coverage](https://codecov.io/gh/snap-stanford/deepsnap/branch/master/graph/badge.svg)](https://codecov.io/github/snap-stanford/deepsnap?branch=master)
[![Downloads](https://pepy.tech/badge/deepsnap)](https://pepy.tech/project/deepsnap)
[![Repo size](https://img.shields.io/github/repo-size/snap-stanford/deepsnap?color=yellow)](https://github.com/snap-stanford/deepsnap/archive/refs/heads/master.zip)


**[Documentation](https://snap.stanford.edu/deepsnap/)** | **[Examples](https://github.com/snap-stanford/deepsnap/tree/master/examples)** | **[Colab Notebooks](https://snap.stanford.edu/deepsnap/notes/colab.html)**

DeepSNAP is a Python library to assist efficient deep learning on graphs. 
DeepSNAP features in its support for flexible graph manipulation, standard pipeline, heterogeneous graphs and simple API.

DeepSNAP bridges powerful graph libraries such as [NetworkX](https://networkx.github.io/) and deep learning framework [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest). With an intuitive and easy-than-ever API, DeepSNAP addresses the above pain points:

- DeepSNAP currently supports a NetworkX-based backend (also SnapX-based backend for homogeneous undirected graph), allowing users to seamlessly call hundreds of graph algorithms available to manipulate / transform the graphs, even at every training iteration.
- DeepSNAP provides a standard pipeline for dataset split, negative sampling and defining node/edge/graph-level objectives, which are transparent to users.
- DeepSNAP provides efficient support for flexible and general heterogeneous GNNs, that supports both node and edge heterogeneity, and allows users to control how messages are parameterized and passed.
- DeepSNAP has an easy-to-use API that works seamlessly with existing GNN models / datasets implemented in PyTorch Geometric. There is close to zero learning curve if the user is familiar with PyTorch Geometric.

# Installation
To install the DeepSNAP, ensure [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest) and [NetworkX](https://networkx.github.io/) are installed. Then:


```sh
$ pip install deepsnap
```
Or build from source:
```sh
$ git clone https://github.com/snap-stanford/deepsnap
$ cd deepsnap
$ pip install .
```

# Example

Examples using DeepSNAP are provided within the code repository.

```sh
$ git clone https://github.com/snap-stanford/deepsnap
```

**Node classification**:
```sh
$ cd deepsnap/examples/node_classification # node classification
$ python node_classification_planetoid.py
```

**Link prediction**:
```sh
$ cd deepsnap/examples/link_prediction # link prediction
$ python link_prediction_cora.py
```

**Graph classification**:
```sh
$ cd deepsnap/examples/graph_classification # graph classification
$ python graph_classification_TU.py
```

# Documentation
For comprehensive overview, introduction, tutorial and example, please refer to [Full Documentation](https://snap.stanford.edu/deepsnap/)
