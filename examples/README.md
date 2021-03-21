# Examples

We provide some examples which utilize the DeepSNAP.
The examples include node classification, link prediction and graph classification for not only on homogeneous but also on heterogeneous graphs.

## Basic Examples

### Homogeneous Graph
* [Node classification](node_classification/): Node classification examples on the [Planetoid](https://arxiv.org/abs/1603.08861) dataset, including the citation graphs such as Cora, CiteSeer and PubMed.
* [Link prediction](link_prediction/): Link prediction examples on the datasets such as the Cora and WN18.
* [Graph classification](graph_classification.py): Graph classification on the DD or Enzymes graph.

### Heterogeneous Graph
* [Node classification](heterogeneous/node_classification.py): Node classification on a concatenated (Cora and Citeseer) graph. It classifies node label for each graph.
* [Link prediction](heterogeneous/link_prediction.py): Link prediction for WordNet by using the heterogeneous GNN. It predicts link for each edge type and treats each edge type prediction as a binary classification task.

## Bio Application
* [Node classification](bio_application): Some bio-related node classification examples.

## Data
Most of data can be downloaded from the PyTorch Geometric, such as the DD, Enzymes, Cora and Citeseer etc.

The WordNet dataset can be downloaded through:
```
wget https://www.dropbox.com/s/qdwi3wh18kcumqd/WN18.gpickle
```

For the `bio_appication` dataset, please refer to [https://www.kaggle.com/farzaan/deepsnap-bio-application-datasets](https://www.kaggle.com/farzaan/deepsnap-bio-application-datasets).
