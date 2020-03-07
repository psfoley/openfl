#!/usr/bin/env python3
from tfedlrn.gpuutils import pick_cuda_device
pick_cuda_device()

import argparse
import os
import logging
import importlib

from tfedlrn import load_yaml, get_object
from models import export_weights
from setup_logging import setup_logging

def get_data(data_names_to_paths, data_name, code_path, class_name, **kwargs):
    data_path = data_names_to_paths[data_name]
    return get_object(code_path, class_name, data_path=data_path, **kwargs)

def load_model(code_path, **kwargs):
    module = importlib.import_module(code_path)
    model = module.get_model(**kwargs)
    return model


def main(plan_name, plan_type, data_config_fname, logging_config_path, logging_default_level):
    setup_logging(path=logging_config_path, default_level=logging_default_level)

    # FIXME: consistent filesystem (#15)
    script_dir = os.path.dirname(os.path.realpath(__file__))
    if plan_type=='fl':
        base_dir = os.path.join(script_dir, 'federations')
    elif plan_type == 'sfl': 
        base_dir = os.path.join(script_dir, 'simulations')
    
    plan_dir = os.path.join(base_dir, 'plans')
    weights_dir = os.path.join(base_dir, 'weights')

    plan = load_yaml(os.path.join(plan_dir, plan_name))
    model_config = plan['model']
    fed_config = plan['federation']
    data_config = plan['data']

    # FIXME: this will ultimately run in a governor environment and should not require any data to work
    # pick the first collaborator to create the data and model (could be any)
    col_id = fed_config['col_ids'][0]
    data_names_to_paths = load_yaml(os.path.join(base_dir, data_config_fname))['collaborators'][col_id]

    data = get_data(data_names_to_paths, **data_config)

    wrapped_model = get_object(data=data, **model_config)
    
    fpath = os.path.join(weights_dir, fed_config['init_model_fname'])
    model_version = fed_config['model_version']

    export_weights(model_name=wrapped_model.__class__.__name__, 
                   version=model_version, 
                   tensor_dict=wrapped_model.get_tensor_dict(False), 
                   fpath=fpath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan_name', '-pn', type=str, required=True)
    parser.add_argument('--plan_type', '-pt', type=str, choices=['fl', 'sfl'], default='fl')
    parser.add_argument('--data_config_fname', '-dc', type=str, default="local_data_config.yaml")
    parser.add_argument('--logging_config_path', '-c', type=str, default="logging.yaml")
    parser.add_argument('--logging_default_level', '-l', type=str, default="info")
    args = parser.parse_args()
    main(**vars(args))
