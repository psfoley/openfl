#!/usr/bin/env python3

# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

import argparse
import os
import logging
import importlib

from tfedlrn.collaborator.collaborator import Collaborator
from tfedlrn.collaborator.collaboratorgpcclient import CollaboratorGRPCClient
from tfedlrn import load_yaml, get_object

from setup_logging import setup_logging


def get_data(data_names_to_paths, data_name, code_path, class_name, **kwargs):
    data_path = data_names_to_paths[data_name]
    return get_object(code_path, class_name, data_path=data_path, **kwargs)

def get_channel(base_dir, col_id, cert_folder, **col_grpc_client_config):
    cert_dir = os.path.join(base_dir, 'certs', cert_folder) 
    return CollaboratorGRPCClient(ca=os.path.join(cert_dir, 'ca.crt'),
                                  certificate=os.path.join(cert_dir, '{}.crt'.format(col_id)),
                                  private_key=os.path.join(cert_dir, '{}.key'.format(col_id)), 
                                  **col_grpc_client_config)

def main(plan, col_id, data_config_fname, logging_config_fname, logging_default_level):
    setup_logging(path=logging_config_fname, default_level=logging_default_level)

    # FIXME: consistent filesystem (#15)
    script_dir = os.path.dirname(os.path.realpath(__file__))
    base_dir = os.path.join(script_dir, 'federations')
    plan_dir = os.path.join(base_dir, 'plans')

    flplan = load_yaml(os.path.join(plan_dir, plan))
    col_config = flplan['collaborator']
    model_config = flplan['model']
    data_config = flplan['data']
    data_names_to_paths = load_yaml(os.path.join(base_dir, data_config_fname))['collaborators'][col_id]

    col_grpc_client_config = flplan['grpc']
    
    channel = get_channel(base_dir=base_dir, 
                          col_id=col_id,
                          **col_grpc_client_config)

    data = get_data(data_names_to_paths, **data_config)

    wrapped_model = get_object(data=data, **model_config)


    collaborator = Collaborator(col_id=col_id,
                                wrapped_model=wrapped_model, 
                                channel=channel, 
                                **col_config)


    collaborator.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', '-p', type=str, required=True)
    parser.add_argument('--col_id', '-col', type=str, required=True)
    parser.add_argument('--data_config_fname', '-dc', type=str, default="local_data_config.yaml")
    parser.add_argument('--logging_config_fname', '-lc', type=str, default="logging.yaml")
    parser.add_argument('--logging_default_level', '-l', type=str, default="info")
    args = parser.parse_args()
    main(**vars(args))
