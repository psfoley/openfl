# Copyright (C) 2020 Intel Corporation
# Licensed subject to Collaboration Agreement dated February 28th, 2020 between Intel Corporation and Trustees of the University of Pennsylvania.

from math import ceil
import numpy as np

from models.data.fldata import FLData

class KerasFLDataNoPrefetch(FLData):

    def __init__(self, data_path, **kwargs):
        """
        Instantiate the data object 
        (should define numpy arrays:
         self.X_train, self.y_train, self.X_val, and self.y_val
         with data instances enumerated along first axis)

        Returns
        -------
        None
        """
        raise NotImplementedError

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
        return self._get_batch_generator('train', batch_size=None)
    
    def get_val_loader(self):
        """
        Get validation data loader 

        Returns
        -------
        loader object
        """
        return self._get_batch_generator('val', batch_size=None)

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

    def _get_batch_generator(self, train_or_val, batch_size):
        if batch_size == None:
            batch_size = self.batch_size

        if train_or_val == 'train':
            X = self.X_train
            y = self.y_train
        elif train_or_val == 'val':
            X = self.X_val
            y = self.y_val
        else:
            raise ValueError('dtype needs to be train or val')

        # shuffle data indices
        idxs = np.random.permutation(np.arange(X.shape[0]))

        # compute the number of batches
        num_batches = ceil(X.shape[0] / batch_size)

        # build the generator and return it
        return self._batch_generator(X, y, idxs, batch_size, num_batches)


        