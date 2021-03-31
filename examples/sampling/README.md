# Scale up GNNs

This example uses the [Neighbor Sampling](https://arxiv.org/abs/1706.02216) and vanilla [Cluster-GCN](https://arxiv.org/abs/1905.07953) to demonstrate how to use the DeepSNAP to scale up GNNs.

* [Node classification on Cora (neighbor sampling)](neighbor_sampling.py): Using neighbor sampling on the Cora graph, with different sampling ratios.
* [Node classification on Cora (vanilla Cluster-GCN)](cluster.py): Using vanilla Cluster-GCN on the Cora graph, with three different partition / cluster algorithms.

## Training

Run the following command as an example.

```sh
# Train the models for the vanilla Cluster-GCN with CPU and 50 epochs
python cluster.py --device=cpu --epochs=50
```

## Colab Example

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1rOr-vzrWtnVLhF2CYLbou2acOfjuw_fu?usp=sharing)