We provide some examples which utilize the DeepSNAP.
The examples include node classification, link prediction and graph classification for not only the homogeneous but also the heterogeneous graphs.

## Basic Examples

### Homogeneous Graph
* [Node classification](node_classification_cora_old.py): Node classification on the Cora graph.
* [Link prediction Cora](link_prediction_cora.py): Link prediction on the Cora graph (or multiple ones).
* [Link prediction WordNet](wn_prediction.py): Link prediction on the WordNet graph by using edge features.
* [Graph classification](graph_classification.py): Graph classification on the DD or Enzymes graph.

### Heterogeneous Graph
* [Node classification](heterogeneous/node_classification.py): Node classification on a concatenated (Cora and Citeseer) graph. It classifies node label for each graph.
* [Link prediction](heterogeneous/link_prediction.py): Link prediction for WordNet by using the heterogeneous GNN. It predicts link for each edge type and treats each edge type prediction as a binary classification task.

## Bio Application
* [Node classification](bio_application): Some bio-related node classification examples.

## Data
Most of data can be downloaded from the PyTorch Geometric, such as the DD, Enzymes, Cora and Citeseer.

For the WordNet dataset, you can download through:
```
wget https://www.dropbox.com/s/qdwi3wh18kcumqd/WN18.gpickle
```

For the data of `bio_appication`, you can download through:
```
wget https://www.dropbox.com/s/1lhi0piap2g61m1/data.zip
```
Then unzip and put it into the `bio_application` folder.
