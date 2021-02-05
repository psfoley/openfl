from openfl.utilities import TensorKey
import hashlib
import numpy as np
import tensorflow.keras as keras
from openfl.federated.types import TypeHandler
from copy import deepcopy

class KerasModelTypeHandler(TypeHandler):

    def __init__(self):
        self.send_delta = True
        self.compress = True
        self.aggregation_type = 'weighted_mean'


    @staticmethod
    def attr_to_map(attribute,attribute_name,round_phase='end',round_num=0, report=False, origin='LOCAL'):
        """Transform the attribute to a {TensorKey: nparray} map for transport."""

        tensorkey_nparray_dict = {}
        weight_names = [TensorKey(weight.name, origin, round_num, round_phase, report, (f'{attribute_name}',)) 
            for weight in attribute.weights]
        weight_values = attribute.get_weights()
        for name, value in zip(weight_names, weight_values):
            tensorkey_nparray_dict[name] = value

        return tensorkey_nparray_dict
    

    @staticmethod
    def _set_weights_dict(obj, weights_dict):
        """Set the object weights with a dictionary.

        The obj can be a model or an optimizer.

        Args:
            obj (Model or Optimizer): The target object that we want to set
            the weights.
            weights_dict (dict): The weight dictionary.

        Returns:
            None
        """
        weight_names = [weight.name for weight in obj.weights]
        weight_values = [weights_dict[name] for name in weight_names]
        obj.set_weights(weight_values)

    def map_to_attr(self,attribute,tensorkey_nparray_map):
        """Transform tensorkey map to attribute"""

        # self._set_weights_dict(self.model, tensor_dict)
        # It is possible to pass in opt variables from the input tensor dict
        # This will make sure that the correct layers are updated
        model_weight_names = [weight.name for weight in attribute.weights]
        model_weights_dict = {
            name: tensorkey_nparray_map[name] for name in model_weight_names
        }
        self._set_weights_dict(attribute, model_weights_dict)

        return attribute

    @staticmethod
    def get_tensorkeys(attribute,attribute_name,round_phase='start',round_num=0, report=False, origin='LOCAL'):
        """Get tensorkeys will always be run at the end of the round"""
        tensorkey_map = KerasModelTypeHandler.attr_to_map(attribute, attribute_name,
                round_phase=round_phase, round_num=round_num, report=report, origin=origin)
        return tensorkey_map.keys()

    @staticmethod
    def get_hash(attribute):
        hasher = hashlib.sha384()
        weights = attribute.get_weights()
    
        for layer in weights:
            hasher.update(layer.data.tobytes())
        return hasher.hexdigest()

