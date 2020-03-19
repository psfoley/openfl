# Copyright (C) 2020 Intel Corporation
# Licensed subject to Collaboration Agreement dated February 28th, 2020 between Intel Corporation and Trustees of the University of Pennsylvania.

import yaml
import importlib

def load_yaml(path):
    plan = None
    with open(path, 'r') as f:
        plan = yaml.safe_load(f.read())
    return plan

def get_object(code_path, class_name, **kwargs):
    module = importlib.import_module(code_path)
    return module.__getattribute__(class_name)(**kwargs)
