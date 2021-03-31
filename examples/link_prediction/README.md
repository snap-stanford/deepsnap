# Link Prediction

* [Link prediction on Cora (graph backend)](link_prediction_cora.py): Link prediction on the Cora graph using the NetworkX as the DeepSNAP graph manipulation backend.
* [Link prediction on Cora (tensor backend)](link_prediction_cora_tensor.py): Link prediction on the Cora graph only using tensors as the DeepSNAP graph manipulation backend.
* [Link prediction on WN18 (graph backend)](link_prediction_wn.py): Link prediction on the WN18 dataset using the NetworkX as the DeepSNAP graph manipulation backend.
* [Link prediction on WN18 (tensor backend)](link_prediction_wn_tensor.py): Link prediction on the WN18 dataset only using tensors as the DeepSNAP graph manipulation backend.

## Data
Most of data can be downloaded from the PyTorch Geometric.

The WordNet dataset can be downloaded through:
```
wget https://www.dropbox.com/s/qdwi3wh18kcumqd/WN18.gpickle
```

## Training

Run the following command as an example.

```sh
# Train the model on the multiple Cora graphs using CPU, 200 epochs
python link_prediction_cora_tensor.py --device=cpu --epochs=200 --multigraph
```

## Colab Example

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ycdlJuse7l2De7wi51lFd_nCuaWgVABc?usp=sharing)