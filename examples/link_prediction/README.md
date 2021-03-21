# Link Prediction

* [Link prediction on Cora (graph backend)](link_prediction_cora.py): Link prediction on the Cora graph using the NetworkX as the DeepSNAP graph manipulation backend.
* [Link prediction on Cora (tensor backend)](link_prediction_cora_tensor.py): Link prediction on the Cora graph only using tensors as the DeepSNAP graph manipulation backend.


## Training

Run the following command as an example.

```sh
# Train the model on the multiple Cora graphs using CPU, 200 epochs
python link_prediction_cora_tensor.py --device=cpu --epochs=200 --multigraph
```