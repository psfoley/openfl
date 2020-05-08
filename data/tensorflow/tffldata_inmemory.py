# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

from math import ceil
import numpy as np

from data import FLData

class TensorFlowFLDataInMemory(FLData):

    def __init__(self, batch_size):
        """
        Instantiate the data object 
        
        Returns
        -------
        None
        """
        self.batch_size = batch_size
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None

        # Child classes should have init signature:
        # (self, data_path, batch_size, **kwargs), should call this __init__ and then
        # define self.X_train, self.y_train, self.X_val, and self.y_val

    def get_feature_shape(self):
        """
        get the shape of an example feature array 

        Returns
        -------
        tuple - shape of an example feature array
        """
        return self.X_train[0].shape
    
    def get_train_loader(self, batch_size=None):
        """
        Get training data loader 

        Returns
        -------
        loader object
        """      
        return self._get_batch_generator(X=self.X_train, y=self.y_train, batch_size=batch_size)
    
    def get_val_loader(self, batch_size=None):
        """
        Get validation data loader 

        Returns
        -------
        loader object
        """
        return self._get_batch_generator(X=self.X_val, y=self.y_val, batch_size=batch_size)

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


        