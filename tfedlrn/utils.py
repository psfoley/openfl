# Copyright (C) 2020 Intel Corporation
# Licensed subject to Collaboration Agreement dated February 28th, 2020 between Intel Corporation and Trustees of the University of Pennsylvania.

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
