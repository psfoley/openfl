# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

import numpy as np

from data import FLData


class EmptyData(FLData):
    """A data class with no data
    """

    def __init__(self, data_path, data_size_mean, data_size_std, p_train=0.8, **kwargs):
        """A data class with no data

        """
        n = np.random.normal(loc=data_size_mean, scale=data_size_std)
        self.train_size = int(max(1, n * p_train))
        self.val_size = int(max(1, n - self.train_size))

    def get_feature_shape(self):
        """Returns the shape of the input (feature) array
        """
        return 0  #FIXME: NotImplemented Exception?

    def get_train_loader(self):
        """Returns the training data loader object
        """
        return 0  #FIXME: NotImplemented Exception?

    def get_val_loader(self):
        """Returns the validation data loader object
        """
        return 0  #FIXME: NotImplemented Exception?

    def get_training_data_size(self):
        """Return the size of the training data array
        """
        return self.train_size #FIXME: NotImplemented Exception?

    def get_validation_data_size(self):
        """Return the size of the validation data array
        """
        return self.val_size  #FIXME: NotImplemented Exception?
