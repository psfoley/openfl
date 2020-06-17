# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

import numpy as np

from data import FLData


class RandomData(FLData):

    def __init__(self, feature_shape, label_shape=None, data_path=None, train_batches=1, val_batches=1, batch_size=32, **kwargs):
        if label_shape is None:
            label_shape = feature_shape

        self.batch_size = batch_size
        self.X_train = np.random.random(size=tuple([train_batches * batch_size] + list(feature_shape))).astype(np.float32)
        self.y_train = np.random.random(size=tuple([train_batches * batch_size] + list(label_shape))).astype(np.float32)
        self.X_val = np.random.random(size=tuple([val_batches * batch_size] + list(feature_shape))).astype(np.float32)
        self.y_val = np.random.random(size=tuple([val_batches * batch_size] + list(label_shape))).astype(np.float32)


    def get_feature_shape(self):
        """
        get the shape of an example feature array 

        Returns
        -------
        tuple - shape of an example feature array
        """
        return self.X_train[0].shape
    
    def get_train_loader(self):
        """
        Get training data loader 

        Returns
        -------
        loader object
        """      
        return self._get_batch_generator(X=self.X_train, y=self.y_train, batch_size=self.batch_size)
    
    def get_val_loader(self):
        """
        Get validation data loader 

        Returns
        -------
        loader object
        """
        return self._get_batch_generator(X=self.X_val, y=self.y_val, batch_size=self.batch_size)

    def get_training_data_size(self):
        """
        Get total number of training samples 

        Returns
        -------
        int - number of training samples
        """
        return self.X_train.shape[0]

    def get_validation_data_size(self):
        """
        Get total number of validation samples 

        Returns
        -------
        int - number of validation samples
        """
        return self.X_val.shape[0]

    @staticmethod
    def _batch_generator(X, y, idxs, batch_size, num_batches):
        for i in range(num_batches):
            a = i * batch_size
            b = a + batch_size
            yield X[idxs[a:b]], y[idxs[a:b]]

    def _get_batch_generator(self, X, y, batch_size):
        if batch_size == None:
            batch_size = self.batch_size

        # shuffle data indices
        idxs = np.random.permutation(np.arange(X.shape[0]))

        # compute the number of batches
        num_batches = ceil(X.shape[0] / batch_size)

        # build the generator and return it
        return self._batch_generator(X, y, idxs, batch_size, num_batches)
