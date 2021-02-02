from abc import ABC, abstractmethod

class TypeHandler(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def attr_to_map(attribute,round_phase='end'):
        pass

    @abstractmethod
    def map_to_attr(attribute,tensorkey_nparray_map):
        pass

    @abstractmethod
    def get_tensorkeys(attribute,round_phase='start'):
        pass

    @abstractmethod
    def get_hash(attribute):
        pass
