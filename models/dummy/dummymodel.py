# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

"""Dummy model that sleeps and returns random results
"""
import numpy as np
import time

from models import FLModel


class DummyModel(FLModel):

    def __init__(self, data, layer_shapes, train_time_mean, train_time_std, val_time_mean, val_time_std, **kwargs):
        self.data = data
        self.layer_shapes = layer_shapes
        self.train_time_mean = train_time_mean
        self.train_time_std = train_time_std
        self.val_time_mean = val_time_mean
        self.val_time_std = val_time_std
 
    def train_epoch(self):
        self.random_sleep(self.train_time_mean, self.train_time_std)
        return np.random.random()

    def validate(self):
        self.random_sleep(self.val_time_mean, self.val_time_std)
        return np.random.random()

    def get_tensor_dict(self, with_opt_vars):
        d = {}
        for name, shape in self.layer_shapes.items():
            d[name] = np.random.random(size=tuple(shape)).astype(np.float32)
        return d

    def set_tensor_dict(self, tensor_dict, with_opt_vars):
        pass

    def reset_opt_vars(self):
        pass

    def initialize_globals(self):
        pass

    @staticmethod
    def random_sleep(mean, std):
        t = int(np.random.normal(loc=mean, scale=std))
        t = max(1, t)
        time.sleep(t)
