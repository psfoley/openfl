import tensorflow as tf
from .data_handler import DataHandler

class TensorflowDatasetHandler(DataHandler):

    def __init__(self):
        pass

    def shard_data(self,loader,rank,_size):
        return loader.shard(num_shards=_size, index=rank)
