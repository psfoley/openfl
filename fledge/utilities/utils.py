# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

import numpy as np

def split_tensor_dict_into_floats_and_non_floats(tensor_dict):
    """
    Splits the tensor dictionary into float and non-floating point values

    Splits a tensor dictionary into float and non-float values.

    Args:
        tensor_dict: A dictionary of tensors

    Returns:
        Two dictionaries: the first contains all of the floating point tensors and the second contains all of the non-floating point tensors

    """

    float_dict = {}
    non_float_dict = {}
    for k, v in tensor_dict.items():
        if np.issubdtype(v.dtype, np.floating):
            float_dict[k] = v
        else:
            non_float_dict[k] = v
    return float_dict, non_float_dict

def split_tensor_dict_for_holdouts(logger, tensor_dict, holdout_types = ['non_float'], holdout_tensor_names = []):
    """
    Splits a tensor according to tensor types.

    Args:
        logger: The log object
        tensor_dict: A dictionary of tensors
        holdout_types: A list of types to extract from the dictionary of tensors
        holdout_tensor_names: A list of tensor names to extract from the dictionary of tensors

    Returns:
        Two dictionaries: the first is the original tensor dictionary minus the holdout tenors and the second is a tensor dictionary with only the holdout tensors

    """
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
