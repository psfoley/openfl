from openfl.federated.data.handlers import DataHandler

class NumpyDataLoaderHandler(DataHandler):

    def __init__(self):
        pass

    @staticmethod
    def get_dependencies():
        return ['numpy']

    @staticmethod
    def type():
        import numpy as np
        return np.ndarray

    def shard_data(self,loader,rank,_size):
        return loader[rank::_size]
