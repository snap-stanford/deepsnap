# DeepSNAP

[![PyPI](https://img.shields.io/pypi/v/deepsnap.svg)](https://pypi.org/project/deepsnap/) [![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/snap-stanford/deepsnap/blob/master/LICENSE) ![Build Status](https://travis-ci.org/snap-stanford/deepsnap.svg?branch=master)

[Full Documentation](https://snap.stanford.edu/deepsnap/)

DeepSNAP is a Python library to assist efficient deep learning on graphs. 
DeepSNAP features in its support for flexible graph manipulation, standard pipeline, heterogeneous graphs and simple API.

DeepSNAP bridges powerful graph libraries such as [NetworkX](https://networkx.github.io/) and deep learning framework [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest). With an intuitive and easy-than-ever API, DeepSNAP addresses the above pain points:

- DeepSNAP currently supports a NetworkX-based backend, allowing users to seamlessly call hundreds of graph algorithms available to manipulate / transform the graphs, even at every training iteration. (Look forward to other backends such as Snap.py for future release).
- DeepSNAP provides a standard pipeline for dataset split, negative sampling and defining node/edge/graph-level objectives, which are transparent to users.
- DeepSNAP provides efficient support for flexible and general heterogeneous GNNs, that supports both node and edge heterogeneity, and allows users to control how messages are parameterized and passed.
- DeepSNAP has an easy-to-use API that works seamlessly with existing GNN models / datasets implemented in PyTorch Geometric. There is close to zero learning curve if the user is familiar with PyTorch Geometric.

# Installation
To install the DeepSNAP, ensure [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest) and [NetworkX](https://networkx.github.io/) are installed. Then:


```sh
$ pip install deepsnap
```

# Example
Examples for using DeepSNAP are provided with code repository.

```sh
$ git clone https://github.com/snap-stanford/deepsnap
$ cd deepsnap/examples
$ python node_classification_cora.py # node classification
$ python link_prediction_cora.py # link prediction
$ python graph_classification_TU.py # graph classification
```


# Documentation
For comprehensive overview, introduction, tutorial and example, please refer to [Full Documentation](https://snap.stanford.edu/deepsnap/)
