from openfl.federated.data.handlers import DataHandler

class TensorflowDatasetHandler(DataHandler):

    def __init__(self):
        pass

    @staticmethod
    def get_dependencies():
        return ['tensorflow']

    @staticmethod
    def type():
        from tensorflow.data import Dataset
        return Dataset

    def shard_data(self,loader,rank,_size):
        return loader.shard(num_shards=_size, index=rank)
