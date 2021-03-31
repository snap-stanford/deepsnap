# Heterogeneous Graph Node Classification

* [Node classification on a synthesized heterogeneous graph (graph backend)](node_classification_syn.py): Node classification on a synthesized heterogeneous graph. We concatenate Cora and Citeseer graphs into one heterogeneous graph. Nodes and edges in each graph will be treated  This example uses the NetworkX as the DeepSNAP graph manipulation backend.
* [Node classification on a synthesized heterogeneous graph (tensor backend)](node_classification_syn_tensor.py): Similar to the graph backend example, this is an example on the synthesized heterogeneous graph. But the DeepSNAP graph manipulation backend uses only the tensors.
* [Node classification on the ACM dataset (tensor backend)](node_classification_acm.py): Heterogeneous node classification on the ACM dataset ([Wang et al. (2019)](https://arxiv.org/abs/1903.07293)). We use DeepSNAP built-in `HeteroSAGEConv` as the GNN layer. This example trains two models. One uses the mean aggregation for message types and another one uses the semantic level attention proposed in **HAN** ([Wang et al. (2019)](https://arxiv.org/abs/1903.07293)). To make the training faster, this example uses the tensor backend and the `edge_index` is in `torch_sparse.SparseTensor` format.

# Data

The used ACM dataset here is proposed in **HAN** ([Wang et al. (2019)](https://arxiv.org/abs/1903.07293)) and we extracted from [DGL](https://www.dgl.ai/)'s [ACM.mat](https://data.dgl.ai/dataset/ACM.mat). It can be downloaded through:
```
wget https://www.dropbox.com/s/8c3102hm4ffm092/acm.pkl
```

## Training

Run the following command as an example.

```sh
python node_classification_acm.py
```

## Colab Example

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1L-0kaLqeiT6lHhjHxAzP5sHIcb4b4e7G?usp=sharing)