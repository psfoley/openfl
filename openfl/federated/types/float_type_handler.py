from openfl.utilities import TensorKey
import hashlib
import numpy as np
from openfl.federated.types import TypeHandler
from copy import deepcopy

class FloatTypeHandler(TypeHandler):

    def __init__(self):
        self.send_delta = False
        self.compress = False
        self.aggregation_type = 'weighted_mean'

    @staticmethod
    def get_dependencies():
        """What are the dependencies for this type?"""
        return []

    @staticmethod
    def type():
        """The type that this class handles"""
        return float

    @staticmethod
    def attr_to_map(attribute,attribute_name, round_phase='end',round_num=0, report=False, origin='LOCAL'):
        """Transform the attribute to a {TensorKey: nparray} map for transport."""

        tensorkey_nparray_dict = {
                TensorKey(attribute_name, origin, round_num, round_phase, report, (f'{attribute_name}',)):
                np.array(attribute)
        }

        return tensorkey_nparray_dict


    @staticmethod
    def map_to_attr(attribute,tensorkey_nparray_map):
        """Transform tensorkey map to attribute"""
        return float(list(tensorkey_nparray_map.values())[0])

    @staticmethod
    def get_tensorkeys(attribute,attribute_name,round_phase='start',round_num=0, report=False, origin='LOCAL'):
        """Get tensorkeys will always be run at the end of the round"""
        tensorkey_map = FloatTypeHandler.attr_to_map(attribute, attribute_name,
                round_phase=round_phase, round_num=round_num, report=report, origin=origin)
        return tensorkey_map.keys()

    @staticmethod
    def get_hash(attribute):
        hasher = hashlib.sha384()
        hasher.update(str(attribute).encoding('utf-8'))
        return hasher.hexdigest()

