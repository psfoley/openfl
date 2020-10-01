# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

from fledge.federated import TensorFlowDataLoader

from .mnist_utils import load_mnist_shard

class TensorFlowMNISTInMemory(TensorFlowDataLoader):
    """
    TensorFlow Data Loader for MNIST Dataset
    """

    def __init__(self, data_path, batch_size, **kwargs):
        """
        Initializer

        Args:
            data_path: File path for the dataset
            batch_size (int): The batch size for the data loader
            **kwargs: Additional arguments, passed to super init and load_mnist_shard
        """

        super().__init__(batch_size, **kwargs)

        # TODO: We should be downloading the dataset shard into a directory
        # TODO: There needs to be a method to ask how many collaborators and what index/rank is this collaborator.
        # Then we have a way to automatically shard based on rank and size of collaborator list.
        data_path = 1 # Hard-coding this for now to ignore path in plan, but update me once we have rank/size metrics

        _, num_classes, X_train, y_train, X_valid, y_valid = load_mnist_shard(shard_num = data_path, **kwargs)

        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid

        self.num_classes = num_classes