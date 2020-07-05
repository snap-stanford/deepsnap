#!/usr/bin/env bash

# apply ego to each batch
python graph_classification_TU_transform.py --transform_batch ego --radius 1
python graph_classification_TU_transform.py --transform_batch ego --radius 2
python graph_classification_TU_transform.py --transform_batch ego --radius 3
python graph_classification_TU_transform.py --transform_batch ego --radius 4
python graph_classification_TU_transform.py --transform_batch ego --radius 5
# apply ego to whole dataset
python graph_classification_TU_transform.py --transform_dataset ego --radius 1
python graph_classification_TU_transform.py --transform_dataset ego --radius 2
python graph_classification_TU_transform.py --transform_dataset ego --radius 3
python graph_classification_TU_transform.py --transform_dataset ego --radius 4
python graph_classification_TU_transform.py --transform_dataset ego --radius 5

# apply shortest path to each batch
python graph_classification_TU_transform.py --transform_batch path
# apply shortest path to whole dataset
python graph_classification_TU_transform.py --transform_dataset path