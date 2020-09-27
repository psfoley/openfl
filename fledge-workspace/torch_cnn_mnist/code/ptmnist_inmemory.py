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
            **kwargs: Additional arguments, passed to super init and load_mnist_shard
        """
        super().__init__(batch_size, **kwargs)

        # TODO: We should be downloading the dataset shard into a directory
        # TODO: There needs to be a method to ask how many collaborators and what index/rank is this collaborator.
        # Then we have a way to automatically shard based on rank and size of collaborator list.
        data_path = 1 # Hard-coding this for now to ignore the data path in plan, but update me once we have rank/size metrics

        _, num_classes, X_train, y_train, X_val, y_val = load_mnist_shard(shard_num=data_path, **kwargs)

        self.training_data_size = len(X_train)
        self.validation_data_size = len(X_val)
        self.num_classes = num_classes
        self.train_loader = self.create_loader(X=X_train, y=y_train, shuffle=True)
        self.val_loader = self.create_loader(X=X_val, y=y_val, shuffle=False)
    
    



        
