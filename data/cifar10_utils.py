# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

import numpy as np
from math import ceil

# FIXME: we should remove the keras dependency since it is only really for file downloading
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.python.keras.utils.data_utils import get_file
from tensorflow.keras.datasets import cifar10


def _load_raw_datashards(shard_num, nb_collaborators):
    #origin_link = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    #hash_value = 'c58f30108f718f92721af3b95e74349a'
    #path = get_file('cifar10.tar.gz', origin=origin_link, file_hash=hash_value)
    img_rows, img_cols, img_channel = 32, 32, 3
    num_classes = 10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # fix the label dimension to be (N,)
    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)
    
    # create the shards
    X_train_shards = x_train[shard_num::nb_collaborators]
    y_train_shards = y_train[shard_num::nb_collaborators]
    
    X_test_shards = x_test[shard_num::nb_collaborators]
    y_test_shards  = y_test[shard_num::nb_collaborators]
    return (X_train_shards, y_train_shards), (X_test_shards, y_test_shards)

def load_cifar10_shard(shard_num, nb_collaborators, categorical=True, channels_last=False, **kwargs):
    """
    Load the CIFAR10 dataset.

    Params
    ------
    raw_path: str
        The path to the raw npz file.

    Returns
    -------
    list
        The input shape.
    int
        The number of classes.
    numpy.ndarray
        The training data.
    numpy.ndarray
        The training labels.
    numpy.ndarray
        The validation data.
    numpy.ndarray
        The validation labels.
    """
    img_rows, img_cols, img_channel = 32, 32, 3
    num_classes = 10

    (X_train, y_train), (X_test, y_test) = _load_raw_datashards(shard_num, nb_collaborators)

    if channels_last:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, img_channel)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, img_channel)
        input_shape = (img_rows, img_cols, img_channel)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_channel, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], img_channel, img_rows, img_cols)
        input_shape = (img_channel, img_rows, img_cols)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    if categorical:
        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)

    return input_shape, num_classes, X_train, y_train, X_test, y_test

