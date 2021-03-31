# Heterogeneous Graph Link Prediction

* [Link prediction on WN18 (graph backend)](link_prediction.py): Link prediction on the WN18 dataset using the NetworkX as the DeepSNAP graph manipulation backend.
* [Link prediction on WN18 (tensor backend)](link_prediction_tensor.py): Link prediction on the WN18 dataset only using tensors as the DeepSNAP graph manipulation backend.

## Data
Most of data can be downloaded from the PyTorch Geometric.

The WordNet dataset can be downloaded through:
```
wget https://www.dropbox.com/s/qdwi3wh18kcumqd/WN18.gpickle
```

## Training

Run the following command as an example.

```sh
# Train the model using CUDA and 200 epochs
python link_prediction.py --epochs=200
```

## Colab Example

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1wVGUfUno5Kgs2H-jEGFcm0EogN7DEd-w?usp=sharing)