import numpy as np
from torch.utils.data import DataLoader
from openfl.federated.data.handlers import PyTorchDataLoaderHandler, NumpyDataLoaderHandler

class DataHandlerFactory:

    def __init__(self):
        pass

    def is_supported(self,attr):
        """Does the attribute have a type handler?"""
        if isinstance(attr,(DataLoader,np.ndarray)):
            return True
        return False

    def get_data_handler(self,attr):
        if isinstance(attr,DataLoader):
            return PyTorchDataLoaderHandler()
        elif isinstance(attr,np.ndarray):
            return NumpyDataLoaderHandler()
        else:
            raise ValueError(f'{type(attr)} does not have a supported DataHandler')


