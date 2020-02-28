#!/usr/bin/env python3
import argparse
import os
import logging
import importlib

from tfedlrn import load_yaml
from single_proc_fed import federate
from setup_logging import setup_logging

def load_func(code_path, func_name):
    module = importlib.import_module(code_path)
    function = module.__getattribute__(func_name)
    return function

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
    data_config = load_yaml(os.path.join(base_dir, data_config_fname))['collaborators']
    agg_config = sflplan['aggregator']
    col_config = sflplan['collaborators']
    model_config = sflplan['model']
    fed_config = sflplan['federation']

    # determine filepaths for model protobufs
    init_model_fpath = os.path.join(weights_dir, fed_config['initial_weights'])
    latest_model_fpath = os.path.join(weights_dir, fed_config['latest_weights'])
    best_model_fpath = os.path.join(weights_dir, fed_config['best_weights'])
 

    # get the BraTS data objects for each collaborator
    col_ids = col_config['col_ids']
    get_data_funcs = {col_id: load_func(data_config[col_id]['code_path'], \
                                        data_config[col_id]['data_object_name'])\
                      for col_id in col_ids}
    data_paths = {col_id: data_config[col_id]['data_path'] for col_id in col_ids}
    data_kwargs = {col_id: data_config[col_id]['data_kwargs'] for col_id in col_ids}
    col_data = {col_id: get_data_funcs[col_id](data_path=data_paths[col_id], 
                                               **data_kwargs[col_id])\
                for col_id in col_ids}

    # get the get_model function
    get_model_func = load_func(model_config['code_path'], 'get_model')
    
    # TODO: Run a loop here over various parameter values and iterations
    # TODO: implement more than just saving init, best, and latest model
    federate(get_model_func=get_model_func,
             col_config=col_config, 
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
    parser.add_argument('--data_config_fname', '-dc', type=str, default="data.yaml")
    parser.add_argument('--logging_config_fname', '-c', type=str, default="logging.yaml")
    parser.add_argument('--logging_default_level', '-l', type=str, default="info")
    args = parser.parse_args()
    main(**vars(args))
