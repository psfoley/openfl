#!/usr/bin/env python3
import argparse
import os
import logging
import importlib

from tfedlrn import load_flplan
from models import export_weights
from setup_logging import setup_logging


def load_model(code_path, **kwargs):
    module = importlib.import_module(code_path)
    model = module.get_model(**kwargs)
    return model


def main(plan, logging_config_path, logging_default_level):
    setup_logging(path=logging_config_path, default_level=logging_default_level)

    # FIXME: consistent filesystem (#15)
    script_dir = os.path.dirname(os.path.realpath(__file__))
    base_dir = os.path.join(script_dir, 'federations')
    plan_dir = os.path.join(base_dir, 'plans')
    weights_dir = os.path.join(base_dir, 'weights')

    flplan = load_flplan(os.path.join(plan_dir, plan))
    model_config = flplan['model']
    agg_config = flplan['aggregator']

    fpath = os.path.join(weights_dir, agg_config['initial_weights'])

    # FIXME: model loading needs to received model paramters from flplan (#17)
    model_config['params']['dataset_path'] = None
    wrapped_model = load_model(model_config['code_path'], **model_config['params'])

    # FIXME: should version match the plan?
    export_weights(wrapped_model.__class__.__name__, 0, wrapped_model.get_tensor_dict(False), fpath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', '-p', type=str, required=True)
    parser.add_argument('--logging_config_path', '-c', type=str, default="logging.yaml")
    parser.add_argument('--logging_default_level', '-l', type=str, default="info")
    args = parser.parse_args()
    main(**vars(args))
