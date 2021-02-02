from openfl.utilities import TensorKey
import hashlib
import numpy as np
import torch
from openfl.federated.types import TypeHandler
from copy import deepcopy

class PyTorchModuleTypeHandler(TypeHandler):

    def __init__(self):
        self.send_delta = True
        self.compress = True
        self.aggregation_type = 'weighted_mean'


    @staticmethod
    def attr_to_map(attribute,attribute_name,round_phase='end',round_num=0, report=False, origin='LOCAL'):
        """Transform the attribute to a {TensorKey: nparray} map for transport."""
        state_dict = attribute.state_dict()
        # deep copy so as to decouple from active model
        state = deepcopy(state_dict)
    
        for k, v in state.items():
            # When restoring, we currently assume all values are tensors.
            if not torch.is_tensor(v):
                raise ValueError('We do not currently support non-tensors '
                                 'coming from model.state_dict()')
            # get as a numpy array, making sure is on cpu
            state[k] = v.cpu().numpy()

        tensorkey_nparray_dict = {
                TensorKey(tensor_name, origin, round_num, round_phase, report, (f'{attribute_name}',)):
                nparray for tensor_name, nparray in state.items()
        }

        return tensorkey_nparray_dict


    @staticmethod
    def map_to_attr(attribute,tensorkey_nparray_map):
        """Transform tensorkey map to attribute"""
        state_dict = {}
        for k in tensorkey_nparray_map:
            state_dict[k] = torch.from_numpy(tensorkey_nparray_map[k])
        attribute.load_state_dict(state_dict)
        print(f'New model: {attribute}')
        return attribute

    @staticmethod
    def get_tensorkeys(attribute,attribute_name,round_phase='start',round_num=0, report=False, origin='LOCAL'):
        """Get tensorkeys will always be run at the end of the round"""
        tensorkey_map = PyTorchModuleTypeHandler.attr_to_map(attribute, attribute_name,
                round_phase=round_phase, round_num=round_num, report=report, origin=origin)
        return tensorkey_map.keys()

    @staticmethod
    def get_hash(attribute):
        hasher = hashlib.sha384()
        state_dict = attribute.state_dict()
        state = deepcopy(state_dict)
    
        for k, v in state.items():
            # When restoring, we currently assume all values are tensors.
            if not torch.is_tensor(v):
                raise ValueError('We do not currently support non-tensors '
                                 'coming from model.state_dict()')
            # get as a numpy array, making sure is on cpu
            state[k] = v.cpu().numpy()

        for layer in state.values():
            hasher.update(layer.data.tobytes())
        return hasher.hexdigest()

