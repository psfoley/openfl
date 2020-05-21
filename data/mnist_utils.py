# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.


import numpy as np
# FIXME: we should remove the keras dependency since it is only really for file downloading
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.python.keras.utils.data_utils import get_file


def _load_raw_datashards(shard_num, nb_collaborators):
    origin_folder = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/'
    path = get_file('mnist.npz',
                    origin=origin_folder + 'mnist.npz',
                    file_hash='731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1')
    with np.load(path) as f:
        # get all of mnist
        X_train_tot = f['x_train']
        y_train_tot = f['y_train']
        
        X_test_tot = f['x_test']
        y_test_tot = f['y_test']

    # create the shards
    X_train = X_train_tot[shard_num::nb_collaborators]
    y_train = y_train_tot[shard_num::nb_collaborators]
    
    X_test = X_test_tot[shard_num::nb_collaborators]
    y_test = y_test_tot[shard_num::nb_collaborators]

    return (X_train, y_train), (X_test, y_test)


def load_mnist_shard(shard_num, nb_collaborators, categorical=True, channels_last=True,  **kwargs):
    """
    Load the MNIST dataset.

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
    img_rows, img_cols = 28, 28
    num_classes = 10

    (X_train, y_train), (X_test, y_test) = _load_raw_datashards(shard_num, nb_collaborators)

    if channels_last:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
    else:
        X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print('X_train shape:', X_train.shape)
    print('y_train shape:', y_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    if categorical:
        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)

    return input_shape, num_classes, X_train, y_train, X_test, y_test

