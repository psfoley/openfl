# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

import numpy as np
import torch
import torch.utils.data

from data.fldata import FLData


class PyTorchFLDataInMemory(FLData):

    def __init__(self, batch_size):
        """
        Instantiate the data object 

        Returns
        -------
        None
        """
        self.batch_size = batch_size
        self.train_loader = None
        self.val_loader = None

        # Child classes should have init signature:
        # (self, data_path, batch_size, **kwargs), should call this __init__ and then
        # define loaders: self.train_loader and self.val_loader using the 
        # self.create_loader provided here.

    def get_feature_shape(self):
        """
        get the shape of an example feature array 

        Returns
        -------
        tuple - shape of an example feature array
        """
        return tuple(self.train_loader.dataset[0][0].shape)

    def get_train_loader(self):
        """
        Get training data loader 

        Returns
        -------
        loader object (class defined by inheritor)
        """
        return self.train_loader
    
    def get_val_loader(self):
        """
        Get validation data loader 

        Returns
        -------
        loader object (class defined by inheritor)
        """
        # TODO: Do we want to be able to modify batch size here?
        # If so will have to decide whether to replace the loader.
        return self.val_loader

    def get_training_data_size(self):
        """
        Get total number of training samples 

        Returns
        -------
        int - number of training samples
        """
        # TODO: Do we want to be able to modify batch size here?
        # If so will have to decide whether to replace the loader.
        return len(self.train_loader)

    def get_validation_data_size(self):
        """
        Get total number of validation samples 

        Returns
        -------
        int - number of validation samples
        """
        return len(self.val_loader)


    def create_loader(self, X, y):
        if isinstance(X[0], np.ndarray):
            tX = torch.stack([torch.Tensor(i) for i in X])
        else:
            tX = torch.Tensor(X)
        if isinstance(y[0], np.ndarray):
            ty = torch.stack([torch.Tensor(i) for i in y])
        else:
            ty = torch.Tensor(y)
        return torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(tX, ty), 
                                           batch_size=self.batch_size)




        