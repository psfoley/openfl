#!/usr/bin/env python3

# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

import argparse
import os
import logging

from tfedlrn.aggregator.aggregator import Aggregator
from tfedlrn.aggregator.aggregatorgrpcserver import AggregatorGRPCServer
from tfedlrn import load_yaml 

from setup_logging import setup_logging


def main(plan, logging_config_path, logging_default_level):
    setup_logging(path=logging_config_path, default_level=logging_default_level)

    # FIXME: consistent filesystem (#15)
    script_dir = os.path.dirname(os.path.realpath(__file__))
    base_dir = os.path.join(script_dir, 'federations')
    plan_dir = os.path.join(base_dir, 'plans')
    weights_dir = os.path.join(base_dir, 'weights')

    flplan = load_yaml(os.path.join(plan_dir, plan))
    agg_config = flplan['aggregator']
    fed_config = flplan['federation']
    grpc_server_config = flplan['grpc']

    init_model_fpath = os.path.join(weights_dir, fed_config['init_model_fname'])
    latest_model_fpath = os.path.join(weights_dir, fed_config['latest_model_fname'])
    best_model_fpath = os.path.join(weights_dir, fed_config['best_model_fname'])
    
    agg = Aggregator(init_model_fpath=init_model_fpath,
                     latest_model_fpath=latest_model_fpath,
                     best_model_fpath=best_model_fpath, 
                     **agg_config)

    cert_dir = os.path.join(base_dir, 'certs', grpc_server_config['cert_folder'])

    agg_grpc_server = AggregatorGRPCServer(agg)
    agg_grpc_server.serve(ca=os.path.join(cert_dir, 'ca.crt'),
                          certificate=os.path.join(cert_dir, '{}.crt'.format(agg_config['agg_id'])),
                          private_key=os.path.join(cert_dir, '{}.key'.format(agg_config['agg_id'])), 
                          **grpc_server_config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', '-p', type=str, required=True)
    parser.add_argument('--logging_config_path', '-c', type=str, default="logging.yaml")
    parser.add_argument('--logging_default_level', '-l', type=str, default="info")
    args = parser.parse_args()
    main(**vars(args))
