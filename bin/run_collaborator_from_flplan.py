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

def get_channel(base_dir, cert_common_name, **col_grpc_client_config):
    cert_dir = os.path.join(base_dir, col_grpc_client_config.pop('cert_folder', 'pki')) # default to 'pki

    return CollaboratorGRPCClient(ca=os.path.join(cert_dir, 'cert_chain.crt'),
                                  certificate=os.path.join(cert_dir, 'col_{}'.format(cert_common_name), 'col_{}.crt'.format(cert_common_name)),
                                  private_key=os.path.join(cert_dir, 'col_{}'.format(cert_common_name), 'col_{}.key'.format(cert_common_name)), 
                                  **col_grpc_client_config)

def main(plan, col_id, cert_common_name, data_config_fname, logging_config_fname, logging_default_level):
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
    
    if cert_common_name is None:
        cert_common_name = col_id

    channel = get_channel(base_dir=base_dir, 
                          cert_common_name=cert_common_name,
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
    parser.add_argument('--cert_common_name', '-ccn', type=str, default=None)
    parser.add_argument('--data_config_fname', '-dc', type=str, default="local_data_config.yaml")
    parser.add_argument('--logging_config_fname', '-lc', type=str, default="logging.yaml")
    parser.add_argument('--logging_default_level', '-l', type=str, default="info")
    args = parser.parse_args()
    main(**vars(args))
