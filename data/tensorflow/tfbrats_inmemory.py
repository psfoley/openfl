# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.


from data import load_from_NIfTY
from data.tensorflow.tffldata_inmemory import TensorFlowFLDataInMemory


class TensorFlowBratsInMemory(TensorFlowFLDataInMemory):
    """TensorFlow Data Loader for the BraTS dataset
    """

    def __init__(self, data_path, batch_size, percent_train=0.8, pre_split_shuffle=True, **kwargs):
        """Initializer

        Args:
            data_path: The file path for the BraTS dataset
            batch_size(int): The batch size to use
            percent_train(float): The percentage of the data to use for training (Default=0.8)
            pre_split_shuffle(boolean): True= shuffle the dataset before performing the train/validate split (Default=True)
            **kwargs: Additional arguments to pass to function

        Returns:
            Data loader with BraTS data
        """

        super().__init__(batch_size)

        X_train, y_train, X_val, y_val = load_from_NIfTY(parent_dir=data_path,
                                                         percent_train=percent_train,
                                                         shuffle=pre_split_shuffle,
                                                         **kwargs)
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
