# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

import numpy as np

from data.tensorflow.tffldata_inmemory import TensorFlowFLDataInMemory
from data import load_cifar10_shard

class TensorFlowCIFAR10InMemory(TensorFlowFLDataInMemory):
    """TensorFlow data loader for CIFAR10 dataset
    """

    def __init__(self, data_path, batch_size, **kwargs):
        """Initializer

        Args:
            data_path: The file path for the BraTS dataset
            batch_size (int): The batch size to use
            **kwargs: Additional arguments to pass to function

        Returns:
            Data loader with BraTS data
        """

        super().__init__(batch_size)

        _, num_classes, X_train, y_train, X_val, y_val = load_cifar10_shard(shard_num=data_path, **kwargs)

        self.num_classes = num_classes
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
