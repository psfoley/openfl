# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

from data import load_from_NIfTY
from data.pytorch.ptfldata_inmemory import PyTorchFLDataInMemory


class PyTorchBratsInMemory(PyTorchFLDataInMemory):
    """Loads the BraTS dataset into memory for the PyTorch examples
    """

    def __init__(self,
                 data_path,
                 batch_size,
                 percent_train=0.8,
                 pre_split_shuffle=True,
                 channels_last=False,
                 **kwargs):

        """Initializer

        Args:
            data_path: The file path for the BraTS dataset
            batch_size (int): The batch size to use
            percent_train (float): The percentage of the data to use for training (Default=0.8)
            pre_split_shuffle (bool): True= shuffle the dataset before performing the train/validate split (Default=True)
            channels_last (bool): True= data is channels last (NHWDC) (Default=False)
            **kwargs: Additional arguments to pass to function

        Returns:
            Data loader with BraTS data
        """

        super().__init__(batch_size)

        X_train, y_train, X_val, y_val = load_from_NIfTY(parent_dir=data_path,
                                                        percent_train=percent_train,
                                                        shuffle=pre_split_shuffle,
                                                        channels_last=channels_last,
                                                        **kwargs)
        self.training_data_size = len(X_train)
        self.validation_data_size = len(X_val)
<<<<<<< HEAD
        self.train_loader = self.create_loader(X=X_train, y=y_train, shuffle=True)
        self.val_loader = self.create_loader(X=X_val, y=y_val, shuffle=False)

        


=======
        self.train_loader = self.create_loader(X=X_train, y=y_train)
        self.val_loader = self.create_loader(X=X_val, y=y_val)
>>>>>>> c832d751d7e2b77ca64307c4eab46fa30046e0db
