import numpy as np
from .data_handler import DataHandler

class NumpyDataLoaderHandler(DataHandler):

    def __init__(self):
        pass

    def shard_data(self,loader,rank,_size):
        return loader[rank::_size]
