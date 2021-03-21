# Heterogeneous Graph Node Classification

* [Node classification on a synthesized heterogeneous graph (graph backend)](node_classification.py): Node classification on a synthesized heterogeneous graph. We concatenate Cora and Citeseer graphs into one heterogeneous graph. Nodes and edges in each graph will be treated  This example uses the NetworkX as the DeepSNAP graph manipulation backend.
* [Node classification on a synthesized heterogeneous graph (tensor backend)](node_classification_tensor.py): Similar to the graph backend example, this is an example on the synthesized heterogeneous graph. But the DeepSNAP graph manipulation backend uses only the tensors.

## Training

Run the following command as an example.

```sh
python node_classification.py
```