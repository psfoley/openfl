#!/bin/bash

col_num=$1

# start and end indices for training and validation sets
ts=$((0 + $col_num * 6000))
te=$((6000 + $col_num * 6000))
vs=$((0 + $col_num * 1000))
ve=$((1000 + $col_num * 1000))

mkdir -p ../datasets/mnist_batch 

python3 \
../models/mnist_cnn_keras/prepare_dataset.py \
-ts=$ts \
-te=$te \
-vs=$vs \
-ve=$ve \
--output_path=../datasets/mnist_batch/mnist_batch.npz

python3 run_collaborator_from_flplan.py -p mnist_a.yaml -col col_$col_num

