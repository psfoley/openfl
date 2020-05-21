#!/usr/bin/env python3

# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

import argparse
import os
import logging
import importlib

from tfedlrn import load_yaml, get_object
from single_proc_fed import federate
from setup_logging import setup_logging

def get_data(data_names_to_paths, data_name, code_path, class_name, **kwargs):
    data_path = data_names_to_paths[data_name]
    return get_object(code_path, class_name, data_path=data_path, **kwargs)

def main(plan, data_config_fname, logging_config_fname, logging_default_level, **kwargs):

    setup_logging(path=logging_config_fname, default_level=logging_default_level)

    # FIXME: consistent filesystem (#15)
    # establish location for fl plan as well as 
    # where to get and write model protobufs
    script_dir = os.path.dirname(os.path.realpath(__file__))
    base_dir = os.path.join(script_dir, 'federations')
    plan_dir = os.path.join(base_dir, 'plans')
    weights_dir = os.path.join(base_dir, 'weights')

    # parse configs from flplan
    flplan = load_yaml(os.path.join(plan_dir, plan))
    by_col_data_names_to_paths = load_yaml(os.path.join(base_dir, data_config_fname))['collaborators']
    fed_config = flplan['federation']
    agg_config = flplan['aggregator']
    col_config = flplan['collaborator']
    model_config = flplan['model']
    data_config = flplan['data']


    init_model_fpath = os.path.join(weights_dir, fed_config['init_model_fname'])
    latest_model_fpath = os.path.join(weights_dir, fed_config['latest_model_fname'])
    best_model_fpath = os.path.join(weights_dir, fed_config['best_model_fname'])

    


    # get the BraTS data objects for each collaborator
    col_ids = fed_config['col_ids']
    col_data = {col_id: get_data(by_col_data_names_to_paths[col_id], **data_config) for col_id in col_ids}
    
    # TODO: Run a loop here over various parameter values and iterations
    # TODO: implement more than just saving init, best, and latest model
    federate(col_config=col_config, 
             agg_config=agg_config,
             col_data=col_data, 
             model_config=model_config, 
             fed_config=fed_config, 
             init_model_fpath = init_model_fpath, 
             latest_model_fpath = latest_model_fpath, 
             best_model_fpath = best_model_fpath, 
             **kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', '-p', type=str, required=True)
    parser.add_argument('--data_config_fname', '-dc', type=str, default="local_data_config.yaml")
    parser.add_argument('--logging_config_fname', '-c', type=str, default="logging.yaml")
    parser.add_argument('--logging_default_level', '-l', type=str, default="info")
    args = parser.parse_args()
    main(**vars(args))
