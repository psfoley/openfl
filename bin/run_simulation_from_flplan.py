#!/usr/bin/env python3

# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

import argparse
import os
import logging

from tfedlrn import load_yaml, parse_fl_plan
from single_proc_fed import federate
from setup_logging import setup_logging


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
    flplan = parse_fl_plan(os.path.join(plan_dir, plan))
    by_col_data_names_to_paths = load_yaml(os.path.join(base_dir, data_config_fname))['collaborators']
    agg_config = flplan['aggregator']
    col_config = flplan['collaborator']
    model_config = flplan['model']
    data_config = flplan['data']
    compression_config = flplan.get('compression_pipeline')


    init_model_fpath = os.path.join(weights_dir, agg_config['init_model_fname'])
    latest_model_fpath = os.path.join(weights_dir, agg_config['latest_model_fname'])
    best_model_fpath = os.path.join(weights_dir, agg_config['best_model_fname'])

  
    # TODO: Run a loop here over various parameter values and iterations
    # TODO: implement more than just saving init, best, and latest model
    federate(data_config=data_config, 
             col_config=col_config, 
             agg_config=agg_config,
             model_config=model_config,
             compression_config=compression_config,
             by_col_data_names_to_paths=by_col_data_names_to_paths, 
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
