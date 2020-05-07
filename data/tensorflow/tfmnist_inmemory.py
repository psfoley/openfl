# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

import numpy as np

from data import load_mnist_shard
from data.tensorflow.tffldata_inmemory import TensorFlowFLDataInMemory



class TensorFlowMNISTInMemory(TensorFlowFLDataInMemory):

    def __init__(self, data_path, batch_size, **kwargs):

        super().__init__(batch_size)

        _, num_classes, X_train, y_train, X_val, y_val = load_mnist_shard(shard_num=data_path, **kwargs)
        
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

        self.num_classes = num_classes
        


        