# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

from data import load_mnist_shard
from data.pytorch.ptfldata_inmemory import PyTorchFLDataInMemory


class PyTorchMNISTInMemory(PyTorchFLDataInMemory):
    """PyTorch data loader for MNIST dataset
    """

    def __init__(self, data_path, batch_size, **kwargs):
        """Instantiate the data object

        Args:
            data_path: The file path to the data
            batch_size: The batch size of the data loader
            **kwargs: Additional arguments to pass to the functions
        """
        super().__init__(batch_size)

        _, num_classes, X_train, y_train, X_val, y_val = load_mnist_shard(shard_num=data_path, **kwargs)

        self.training_data_size = len(X_train)
        self.validation_data_size = len(X_val)
        self.num_classes = num_classes
        self.train_loader = self.create_loader(X=X_train, y=y_train, shuffle=True)
        self.val_loader = self.create_loader(X=X_val, y=y_val, shuffle=False)
    
    



        
