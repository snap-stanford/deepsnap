# Scale up GNNs

This example uses the Neighbor Sampling and vanilla Cluster-GCN to demonstrate how to use the DeepSNAP to scale up GNNs.

* [Node classification on Cora (neighbor sampling)](neighbor_sampling.py): Using neighbor sampling on the Cora graph, with different sampling ratios.
* [Node classification on Cora (vanilla Cluster-GCN](cluster.py): Using vanilla Cluster-GCN on the Cora graph, with three different partition / cluster algorithms.

## Training

Run the following command as an example.

```sh
# Train the models for the vanilla Cluster-GCN with CPU and 50 epochs
python cluster.py --device=cpu --epochs=50
```