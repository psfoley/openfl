#!/usr/bin/env python3

# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

import argparse

import numpy as np

from models.pytorch.pt_cnn.pt_cnn import PyTorchCNN
from data.pytorch.ptfldata_inmemory import PyTorchFLDataInMemory
from data.dummy.randomdata import RandomData
from setup_logging import setup_logging


def create_inmemory_data(train_batches, batch_size):
    rdata = RandomData([1, 28, 28], label_shape=[10], train_batches=train_batches, val_batches=1, batch_size=batch_size)
    data = PyTorchFLDataInMemory(batch_size)
    
    X_train = rdata.X_train
    y_train = rdata.y_train
    X_val   = rdata.X_val
    y_val   = rdata.y_val

    data.training_data_size = len(X_train)
    data.validation_data_size = len(X_val)
    data.num_classes = 10
    data.train_loader = data.create_loader(X=X_train, y=y_train)
    data.val_loader = data.create_loader(X=X_val, y=y_val)
    return data


def dicts_equal(d1, d2):
    k1 = list(sorted(d1.keys()))
    if k1 != list(sorted(d2.keys())):
        return False
    
    for k in k1:
        if k == '__opt_state_needed':
            continue
        a1 = d1[k].flatten()
        a2 = d2[k].flatten()
        if not all(a1 == a2):
            return False
    return True


def main(train_batches, batch_size):
    """Basic test of pytorch native save and load.
    -create an MNIST CNN with random data object
    -train an epoch
    -get the tensor dict
    -save the model in native format
    -destroy the model object
    -create an MNIST CNN with random data object
    -get the new tensor dict and ensure it differs from first
    -load the native model
    -get a third tensor dict and ensure it matches

    Args:
        train_batches (int) : number of batches to train
        batch_size (int)    : size of each batch

    """
    # create our random data object
    data = create_inmemory_data(train_batches, batch_size)

    # create an MNIST CNN
    model = PyTorchCNN(data)

    # train an epoch
    model.train_batches(train_batches)

    # get the tensor dict
    d1 = model.get_tensor_dict(with_opt_vars=True)

    # save the model in native format
    model.save_native('tmp.pkl')

    # destroy the model and data objects
    del model
    del data

    # create our random data object
    data = create_inmemory_data(train_batches, batch_size)

    # create an MNIST CNN
    model = PyTorchCNN(data)

    # train an epoch
    model.train_batches(train_batches)

    # get the tensor dict
    d2 = model.get_tensor_dict(with_opt_vars=True)

    # check if dicts are equal
    if dicts_equal(d1, d2):
        print("Failure: two models trained to same weights")
    else:
        print("Success: two models trained to different weights")

    # load the native model
    model.load_native('tmp.pkl')

    # get a third tensor dict
    d3 = model.get_tensor_dict(with_opt_vars=True)

    # check if dicts are equal
    if dicts_equal(d1, d3):
        print("Success: loaded model matches first model")
    else:
        print("Failure: loaded model does NOT match first model")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_batches', '-tb', type=int, default=8)
    parser.add_argument('--batch_size', '-bs', type=int, default=32)
    args = parser.parse_args()
    main(**vars(args))
