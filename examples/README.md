# Examples

We provide some examples which utilize the DeepSNAP.
The examples include node classification, link prediction and graph classification not only for homogeneous but also heterogeneous graphs.

## Basic Examples

### Homogeneous Graph
* [Node classification](node_classification/): Node classification examples on the [Planetoid](https://arxiv.org/abs/1603.08861) dataset, including the citation graphs such as Cora, CiteSeer and PubMed.
* [Link prediction](link_prediction/): Link prediction examples on the datasets such as the Cora and WN18.
* [Graph classification](graph_classification/): Graph classification examples on the [TU](https://chrsmrrs.github.io/datasets/) dataset, including DD and Enzymes etc.
* [Sampling](sampling): Scale up GNNs by [Neighbor Sampling](https://arxiv.org/abs/1706.02216) and vanilla [Cluster-GCN](https://arxiv.org/abs/1905.07953) using the DeepSNAP.

### Heterogeneous Graph
* [Node classification](node_classification_hetero): It contains a node classification example on a concatenated (Cora and Citeseer) graph, which classifies node label for each graph. It also contains a heterogeneous node classification example on the ACM dataset ([Wang et al. (2019)](https://arxiv.org/abs/1903.07293)) by using different message type aggregation methods and DeepSNAP built-in `HeteroSAGEConv` layer.
* [Link prediction](link_prediction_hetero): Link prediction examples for WordNet by using DeepSNAP `HeteroGraph` and DeepSNAP heterogeneous GNN functionalities. The examples predict link for each edge type and treats each edge type prediction as a binary classification task.

## Bio Application
* Some bio-related examples.

## Data
Most of data can be downloaded from the PyTorch Geometric, such as the DD, Enzymes, Cora and Citeseer etc.

The WordNet dataset can be downloaded through:
```
wget https://www.dropbox.com/s/qdwi3wh18kcumqd/WN18.gpickle
```

The ACM dataset can be downloaded through:
```
wget https://www.dropbox.com/s/8c3102hm4ffm092/acm.pkl
```

For the `bio_appication` datasets, please refer to [https://www.kaggle.com/farzaan/deepsnap-bio-application-datasets](https://www.kaggle.com/farzaan/deepsnap-bio-application-datasets).
