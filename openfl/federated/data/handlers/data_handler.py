from abc import ABC, abstractmethod

class DataHandler(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def shard_data(self,loader,rank,_size):
        pass
