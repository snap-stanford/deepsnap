# Node Classification

* [Node classification on Planetoid (graph backend)](node_classification_planetoid.py): Node classification on the Planetoid dataset graphs using the NetworkX or SnapX as the DeepSNAP graph manipulation backend.
* [Node classification on Planetoid (tensor backend)](node_classification_planetoid_tensor.py): Node classification on the Planetoid dataset graphs only using tensors as the DeepSNAP graph manipulation backend.

## Training

Run the following command as an example.

```sh
# Train the model on the Cora graph using CPU, predefined splits and SnapX as the backend library
python node_classification_planetoid.py --device=cpu --netlib=sx --split=fixed
```