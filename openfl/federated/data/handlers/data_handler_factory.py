import numpy as np
from torch.utils.data import DataLoader
from tensorflow.data import Dataset
from openfl.federated.data.handlers import PyTorchDataLoaderHandler, NumpyDataLoaderHandler, TensorflowDatasetHandler

class DataHandlerFactory:

    def __init__(self):
        pass

    def is_supported(self,attr):
        """Does the attribute have a type handler?"""
        if isinstance(attr,(DataLoader,np.ndarray,Dataset)):
            return True
        return False

    def get_data_handler(self,attr):
        if isinstance(attr,DataLoader):
            return PyTorchDataLoaderHandler()
        elif isinstance(attr,np.ndarray):
            return NumpyDataLoaderHandler()
        elif isinstance(attr,Dataset):
            return TensorflowDatasetHandler()
        else:
            raise ValueError(f'{type(attr)} does not have a supported DataHandler')


