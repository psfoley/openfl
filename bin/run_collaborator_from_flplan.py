#!/usr/bin/env python3

# Copyright (C) 2020 Intel Corporation

import argparse
import os
import logging
import importlib

from tfedlrn.collaborator.collaborator import Collaborator
from tfedlrn.collaborator.collaboratorgpcclient import CollaboratorGRPCClient
from tfedlrn import load_flplan

from setup_logging import setup_logging

def load_model(code_path, **kwargs):
    module = importlib.import_module(code_path)
    model = module.get_model(**kwargs)
    return model

def main(plan, collaborator_id, logging_config_path, logging_default_level):
    setup_logging(path=logging_config_path, default_level=logging_default_level)

    # FIXME: consistent filesystem (#15)
    script_dir = os.path.dirname(os.path.realpath(__file__))
    base_dir = os.path.join(script_dir, 'federations')
    plan_dir = os.path.join(base_dir, 'plans')

    flplan = load_flplan(os.path.join(plan_dir, plan))
    agg_config = flplan['aggregator']
    model_config = flplan['model']

    tls_config = flplan['tls']
    cert_dir = os.path.join(base_dir, 'certs', tls_config['cert_folder'])

    #FIXME: model loading needs to received model paramters from flplan (#17)
    wrapped_model = load_model(model_config['code_path'], **model_config['params'])
    opt_treatment = flplan['collaborator']['opt_vars_treatment']
    
    channel = CollaboratorGRPCClient(addr=agg_config['addr'],
                                     port=agg_config['port'],
                                     disable_tls=tls_config['disable'],
                                     disable_client_auth=tls_config['disable_client_auth'],
                                     ca=os.path.join(cert_dir, 'ca.crt'),
                                     certificate=os.path.join(cert_dir, 'local.crt'),
                                     private_key=os.path.join(cert_dir, 'local.key'))

    collaborator = Collaborator(collaborator_id,
                                agg_config['id'],
                                flplan['federation'],
                                wrapped_model,
                                channel,
                                model_config['version'],
                                opt_treatment=opt_treatment)

    collaborator.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', '-p', type=str, required=True)
    parser.add_argument('--collaborator_id', '-col', type=str, required=True)
    parser.add_argument('--logging_config_path', '-c', type=str, default="logging.yaml")
    parser.add_argument('--logging_default_level', '-l', type=str, default="info")
    args = parser.parse_args()
    main(**vars(args))
