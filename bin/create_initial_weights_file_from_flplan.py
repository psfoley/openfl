#!/usr/bin/env python3

# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

import argparse
import os
import logging
import importlib

from tfedlrn import load_yaml, get_object, split_tensor_dict_for_holdouts
from tfedlrn.proto import export_weights
from setup_logging import setup_logging

def get_data(data_names_to_paths, data_name, code_path, class_name, **kwargs):
    data_path = data_names_to_paths[data_name]
    return get_object(code_path, class_name, data_path=data_path, **kwargs)

def load_model(code_path, **kwargs):
    module = importlib.import_module(code_path)
    model = module.get_model(**kwargs)
    return model


def main(plan, data_config_fname, logging_config_path, logging_default_level):
    setup_logging(path=logging_config_path, default_level=logging_default_level)

    logger = logging.getLogger(__name__)

    # FIXME: consistent filesystem (#15)
    script_dir = os.path.dirname(os.path.realpath(__file__))
    base_dir = os.path.join(script_dir, 'federations')
    
    plan_dir = os.path.join(base_dir, 'plans')
    weights_dir = os.path.join(base_dir, 'weights')

    plan = load_yaml(os.path.join(plan_dir, plan))
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

    tensor_dict_split_fn_kwargs = wrapped_model.tensor_dict_split_fn_kwargs or {}
    
    tensor_dict, holdout_params = split_tensor_dict_for_holdouts(logger, 
                                                                 wrapped_model.get_tensor_dict(False), 
                                                                 **tensor_dict_split_fn_kwargs)
    logger.warn('Following paramters omitted from global initial model, '\
                'local initialization will determine values: {}'.format(list(holdout_params.keys())))       

    export_weights(model_name=wrapped_model.__class__.__name__, 
                   version=0, 
                   tensor_dict=tensor_dict,
                   fpath=fpath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', '-p', type=str, required=True)
    parser.add_argument('--data_config_fname', '-dc', type=str, default="local_data_config.yaml")
    parser.add_argument('--logging_config_path', '-c', type=str, default="logging.yaml")
    parser.add_argument('--logging_default_level', '-l', type=str, default="info")
    args = parser.parse_args()
    main(**vars(args))
