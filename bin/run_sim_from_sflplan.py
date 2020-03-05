#!/usr/bin/env python3
from tfedlrn.gpuutils import pick_cuda_device
pick_cuda_device()

import argparse
import os
import logging
import importlib

from tfedlrn import load_yaml get_object
from single_proc_fed import federate
from setup_logging import setup_logging

def get_data(data_names_to_paths, data_name, code_path, class_name, **kwargs):
    data_path = data_names_and_paths[data_name]
    return get_object(code_path, class_name, data_path=data_path, **kwargs)

def main(plan, data_config_fname, logging_config_fname, logging_default_level, **kwargs):

    setup_logging(path=logging_config_fname, default_level=logging_default_level)

    # FIXME: consistent filesystem (#15)
    # establish location for sfl plan as well as 
    # where to get and write model protobufs
    script_dir = os.path.dirname(os.path.realpath(__file__))
    base_dir = os.path.join(script_dir, 'simulations')
    plan_dir = os.path.join(base_dir, 'plans')
    weights_dir = os.path.join(base_dir, 'weights')

    # parse configs from sflplan
    sflplan = load_yaml(os.path.join(plan_dir, plan))
    data_names_to_paths = load_yaml(os.path.join(base_dir, data_config_fname))['collaborators']
    fed_config = sflplan['federation']
    agg_config = sflplan['aggregator']
    col_config = sflplan['collaborator']
    model_config = sflplan['model']
    data_config = sflplan['data']
    


    # get the BraTS data objects for each collaborator
    col_ids = fed_config['collaborator_ids']
    col_data = {col_id: get_data(data_names_to_paths, **data_config) for col_id in col_ids}
    
    # get the get_model function
    get_model_func = load_func(model_config['code_path'], 'get_model')
    
    # TODO: Run a loop here over various parameter values and iterations
    # TODO: implement more than just saving init, best, and latest model
    federate(col_config=col_config, 
             agg_config=agg_config,
             col_data=col_data, 
             model_config=model_config, 
             fed_config=fed_config, 
             weights_dir=weights_dir,
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
