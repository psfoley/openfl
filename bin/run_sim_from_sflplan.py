#!/usr/bin/env python3
import argparse
import os
import logging
import importlib

from tfedlrn import load_flplan
from single_proc_fed import federate
from setup_logging import setup_logging

def load_func(code_path, func_name):
    module = importlib.import_module(code_path)
    function = module.__getattribute__(func_name)
    return function

def main(plan, logging_config_path, logging_default_level, base_data_path, **kwargs):

    setup_logging(path=logging_config_path, default_level=logging_default_level)

    # FIXME: consistent filesystem (#15)
    script_dir = os.path.dirname(os.path.realpath(__file__))
    base_dir = os.path.join(script_dir, 'simulations')
    plan_dir = os.path.join(base_dir, 'plans')
    weights_dir = os.path.join(base_dir, 'weights')

    sflplan = load_flplan(os.path.join(plan_dir, plan))

    agg_config = sflplan['aggregator']
    col_config = sflplan['collaborators']
    data_config = sflplan['data']
    model_config = sflplan['model']
 
    get_data_func = load_func(data_config['code_path'], '__init__', **data_config)
    get_model_func = load_func(model_config['code_path'], 'get_model', **model_config)
    
    # TODO: Run a loop here over various parameter values
    # and iterations
    # TODO: implement more than just saving init, best, and latest model
    federate(get_model_func=get_model_func,
             get_data_func=get_data_func,
             col_config=col_config, 
             agg_config=agg_config,
             data_config=data_config, 
             model_config=model_config, 
             weights_dir=weights_dir,
             base_data_path = base_data_path, 
             **kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', '-p', type=str, required=True)
    parser.add_argument('--base_data_path', '-dp', type=str, \
        default='/raid/datasets/BraTS17/by_institution_NIfTY/')
    parser.add_argument('--logging_config_path', '-c', type=str, default="logging.yaml")
    parser.add_argument('--logging_default_level', '-l', type=str, default="info")
    args = parser.parse_args()
    main(**vars(args))
