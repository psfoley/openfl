# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

import numpy as np
# FIXME: we should remove the keras dependency since it is only really for file downloading
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.python.keras.utils.data_utils import get_file

from data.pytorch.ptfldata_inmemory import PyTorchFLDataInMemory
from data import load_cifar10_shard

class PyTorchCIFAR10InMemory(PyTorchFLDataInMemory):

    def __init__(self, data_path, batch_size, **kwargs):
        """
        Instantiate the data object 

        Returns
        -------
        None
        """
        super().__init__(batch_size)

        _, num_classes, X_train, y_train, X_val, y_val = load_cifar10_shard(shard_num=data_path, **kwargs)

        self.num_classes = num_classes
        self.train_loader = self.create_loader(X=X_train, y=y_train)
        self.val_loader = self.create_loader(X=X_val, y=y_val)
