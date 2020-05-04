# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

import yaml
import importlib
import numpy as np

def load_yaml(path):
    plan = None
    with open(path, 'r') as f:
        plan = yaml.safe_load(f.read())
    return plan

def get_object(code_path, class_name, **kwargs):
    module = importlib.import_module(code_path)
    return module.__getattribute__(class_name)(**kwargs)

def split_tensor_dict_into_floats_and_non_floats(tensor_dict):
    float_dict = {}
    non_float_dict = {}
    for k, v in tensor_dict.items():
        if np.issubdtype(v.dtype, np.floating):
            float_dict[k] = v
        else:
            non_float_dict[k] = v
    return float_dict, non_float_dict


def split_tensor_dict_for_holdouts(logger, tensor_dict, holdout_types=['non_float'], holdout_tensor_names=[]):

    # initialization
    tensors_to_send = tensor_dict.copy()
    holdout_tensors = {} 

    # filter by-name tensors from tensors_to_send and add to holdout_tensors
    # (for ones not already held out becuase of their type)
    for tensor_name in holdout_tensor_names:
        if tensor_name not in holdout_tensors.keys():
            try:
                holdout_tensors[tensor_name] = tensors_to_send.pop(tensor_name)
            except KeyError:
                logger.warn('tried to remove tensor: {} not present in the tensor dict'.format(tensor_name))
                continue 
    
    # filter holdout_types from tensors_to_send and add to holdout_tensors
    for holdout_type in holdout_types:
        if holdout_type == 'non_float':
            # filter non floats from tensors_to_send and add to holdouts
            tensors_to_send, non_float_dict = split_tensor_dict_into_floats_and_non_floats(tensors_to_send)
            holdout_tensors = {**holdout_tensors, **non_float_dict}
        else:
            raise ValueError('{} is not a currently suported parameter type to hold out from a tensor dict'.format(holdout_type))
    
    
    return tensors_to_send, holdout_tensors