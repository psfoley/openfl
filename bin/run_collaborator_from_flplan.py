#!/usr/bin/env python3

# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

import argparse
import sys
import os
import logging
import importlib

from tfedlrn.collaborator.collaborator import Collaborator
from tfedlrn.comms.grpc.collaboratorgrpcclient import CollaboratorGRPCClient
from tfedlrn import parse_fl_plan, get_object, load_yaml
from tfedlrn.tensor_transformation_pipelines import get_compression_pipeline

from setup_logging import setup_logging

def get_data(data_names_to_paths, data_name, module_name, class_name, **kwargs):
    data_path = data_names_to_paths[data_name]
    return get_object(module_name, class_name, data_path=data_path, **kwargs)

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

    flplan = parse_fl_plan(os.path.join(plan_dir, plan))
    col_config = flplan['collaborator']
    model_config = flplan['model']
    data_config = flplan['data']
    data_names_to_paths = load_yaml(os.path.join(base_dir, data_config_fname))['collaborators']
    if col_id not in data_names_to_paths:
        sys.exit("Could not find collaborator id \"{}\" in the local data config file. Please edit \"{}\" to specify the datapaths for this collaborator.".format(col_id, data_config_fname))
    data_names_to_paths = data_names_to_paths[col_id]
    if data_config['data_name'] not in data_names_to_paths:
        sys.exit("Could not find data path for collaborator id \"{}\" and dataset name \"{}\". Please edit \"{}\" to specify the path (or shard) for this collaborator and dataset.".format(col_id, data_config['data_name'], data_config_fname))

    if flplan.get('compression_pipeline') is not None:
        compression_pipeline = get_compression_pipeline(**flplan.get('compression_pipeline'))
    else:
        compression_pipeline = None

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
                                compression_pipeline = compression_pipeline, 
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
