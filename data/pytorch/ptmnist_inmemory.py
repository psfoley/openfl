# Copyright (C) 2020 Intel Corporation
# Licensed subject to Collaboration Agreement dated February 28th, 2020 between Intel Corporation and Trustees of the University of Pennsylvania.

from data import load_mnist_shard
from data.pytorch.ptfldata_inmemory import PyTorchFLDataInMemory


class PyTorchMNISTInMemory(PyTorchFLDataInMemory):

    def __init__(self, data_path, batch_size, **kwargs):
        """
        Instantiate the data object 

        Returns
        -------
        None
        """
        super().__init__(batch_size)

        _, num_classes, X_train, y_train, X_val, y_val = load_mnist_shard(shard_num=data_path, **kwargs)

        self.num_classes = num_classes
        self.train_loader = self.create_loader(X=X_train, y=y_train, **kwargs)
        self.val_loader = self.create_loader(X=X_val, y=y_val, **kwargs)

    



        