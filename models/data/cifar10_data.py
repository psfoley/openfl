# Copyright (C) 2020 Intel Corporation
# Licensed subject to Collaboration Agreement dated February 28th, 2020 between Intel Corporation and Trustees of the University of Pennsylvania.

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
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], img_channel, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], img_channel, img_rows, img_cols)
        input_shape = (img_channel, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, img_channel)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, img_channel)
        input_shape = (img_rows, img_cols, img_channel)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print('y_train shape:', y_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    # create the shards
    print('x_train: ', x_train.shape)
    X_train_shards = x_train[shard_num::nb_collaborators]
    print('X_train_shards: ', X_train_shards.shape)
    y_train_shards = y_train[shard_num::nb_collaborators]
    
    X_test_shards = x_test[shard_num::nb_collaborators]
    y_test_shards  = y_test[shard_num::nb_collaborators]
    return (X_train_shards, y_train_shards), (X_test_shards, y_test_shards)
    #return input_shape, num_classes, x_train, y_train, x_test, y_test

def load_cifar10_shard(shard_num, nb_collaborators, data_format=None, categorical=True, **kwargs):
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
    if data_format is None:
        data_format = K.image_data_format()
    if data_format == 'channels_first':
        X_train = X_train.reshape(X_train.shape[0], img_channel, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], img_channel, img_rows, img_cols)
        input_shape = (img_channel, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, img_channel)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, img_channel)
        input_shape = (img_rows, img_cols, img_channel)

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



class CIFAR10Data(object):

    def __init__(self, data_path, batch_size, **kwargs):

        _, num_classes, X_train, y_train, X_val, y_val = load_cifar10_shard(shard_num=data_path, **kwargs)
        
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

        self.num_classes = num_classes

        self.batch_size=batch_size
        
    def get_feature_shape(self):
        return self.X_train[0].shape
    
    def get_training_data(self):
        return self.X_train, self.y_train
    
    def get_validation_data(self):
        return self.X_val, self.y_val

    def get_training_data_size(self):
        return self.X_train.shape[0]

    def get_validation_data_size(self):
        return self.X_val.shape[0]

if __name__=="__main__":
    print('testing start...')
    path = 0
    batch_size = 100
    tmp_dict = {'nb_collaborators':10}
    #obj = CIFAR10Data(path, batch_size, tmp_dict)
    for idx in range(3):
        print('================================================')
        path = idx
        obj = CIFAR10Data(path, batch_size, nb_collaborators=10)
        print('shape: ', obj.get_feature_shape())
        print('================================================')
    print('testing done ...')
        
