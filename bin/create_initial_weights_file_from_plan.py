#!/usr/bin/env python3
import argparse
import os
import logging
import importlib

from tfedlrn import load_yaml
from models import export_weights
from setup_logging import setup_logging


def load_model(code_path, **kwargs):
    module = importlib.import_module(code_path)
    model = module.get_model(**kwargs)
    return model


def main(plan_name, plan_type, logging_config_path, logging_default_level):
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
    agg_config = plan['aggregator']
    fed_config = plan['federation']
    
    if plan_type == 'fl':
        fpath = os.path.join(weights_dir, agg_config['initial_weights'])
    elif plan_type ==  'sfl':
        fpath = os.path.join(weights_dir, fed_config['initial_weights'])

    # FIXME: model loading needs to received model paramters from flplan (#17)
    model_config['params']['dataset_path'] = None
    wrapped_model = load_model(model_config['code_path'], **model_config['params'])

    print('wrapped model is type: {}'.format(type(wrapped_model)))

    # FIXME: should version match the plan?
    export_weights(wrapped_model.__class__.__name__, 0, wrapped_model.get_tensor_dict(False), fpath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan_name', '-pn', type=str, required=True)
    parser.add_argument('--plan_type', '-pt', type=str, choices=['fl', 'sfl'], default='fl')
    parser.add_argument('--logging_config_path', '-c', type=str, default="logging.yaml")
    parser.add_argument('--logging_default_level', '-l', type=str, default="info")
    args = parser.parse_args()
    main(**vars(args))
