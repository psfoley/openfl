# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

import numpy as np

from data import FLData


class EmptyData(FLData):

    def __init__(self, data_path, data_size_mean, data_size_std, p_train=0.8, **kwargs):
        n = np.random.normal(loc=data_size_mean, scale=data_size_std)
        self.train_size = int(max(1, n * p_train))
        self.val_size = int(max(1, n - self.train_size))

    def get_feature_shape(self):
        return 0

    def get_train_loader(self):
        return 0
    
    def get_val_loader(self):
        return 0

    def get_training_data_size(self):
        return self.train_size

    def get_validation_data_size(self):
        return self.val_size
