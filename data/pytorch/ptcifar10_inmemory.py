# Copyright (C) 2020 Intel Corporation
# Licensed subject to Collaboration Agreement dated February 28th, 2020 between Intel Corporation and Trustees of the University of Pennsylvania.

import numpy as np
# FIXME: we should remove the keras dependency since it is only really for file downloading
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.python.keras.utils.data_utils import get_file

from models.data.pytorch.ptfldata_inmemory import PTFLDataInMemory


class PTCIFAR10InMemory(PTFLDataInMemory):

    def __init__(self, data_path, batch_size, **kwargs):
        """
        Instantiate the data object 

        Returns
        -------
        None
        """
        super(PTMNISTInMemory, self).__init__(batch_size)

        _, num_classes, X_train, y_train, X_val, y_val = load_cifar10_shard(shard_num=data_path, **kwargs)

        self.num_classes = num_classes
        self.train_loader = self.create_loader(self, X=X_train, y=y_train, **kwargs)
        self.val_loader = self.create_loader(self, X=X_val, y=y_val, **kwargs)

    

# should find the following functions from Han's work
def _load_raw_datashards(shard_num, nb_collaborators):
    ---
    ---
    --

    rtrn (X_train, y_train), (X_test, y_test)


def load_cifar10_shard(shard_num, nb_collaborators, data_format=None, categorical=True, **kwargs):
    ---
    ---
    
    rtrn input_shape, num_classes, X_train, y_train, X_test, y_test



        