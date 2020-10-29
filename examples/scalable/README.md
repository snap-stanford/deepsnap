# Description
This folder contains examples of scalable training using DeepSNAP. We will provide an easy to use API for distributed training.

# Current Examples
We provide a sample implementation of distributed GraphSAGE for node classification. For comparison we also provide a single worker implementation for node classification. We provide the same for graph classification.

# Future Examples
Node classification - GraphSAGE & GraphSAINT - multi-GPU and multi-node with batches evenly split between workers by index

Graph classification - GIN - multi-GPU and multi-node with batches evenly split between workers by index

Link prediction - METIS partitioning + GraphSAGE - multi-GPU and multi-node with batches created on each worker within each partition
