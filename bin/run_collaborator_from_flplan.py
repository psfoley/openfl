#!/usr/bin/env python3
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

def main(plan, collaborator_id, data_config_fname, logging_config_fname, logging_default_level):
    setup_logging(path=logging_config_fname, default_level=logging_default_level)

    # FIXME: consistent filesystem (#15)
    script_dir = os.path.dirname(os.path.realpath(__file__))
    base_dir = os.path.join(script_dir, 'federations')
    plan_dir = os.path.join(base_dir, 'plans')

    flplan = load_yaml(os.path.join(plan_dir, plan))
    agg_config = flplan['aggregator']
    col_config = flplan['collaborator']
    model_config = flplan['model']
    data_config = flplan['data']
    data_names_to_paths = load_yaml(os.path.join(base_dir, data_config_fname))['collaborators'][collaborator_id]

    col_grpc_client_config = flplan['tls']
    cert_dir = os.path.join(base_dir, 'certs', col_grpc_client_config['cert_folder'])

    data = get_data(data_names_to_paths, **data_config)

    wrapped_model = get_object(data=data, **model_config)

    channel = CollaboratorGRPCClient(ca=os.path.join(cert_dir, 'ca.crt'),
                                     certificate=os.path.join(cert_dir, 'local.crt'),
                                     private_key=os.path.join(cert_dir, 'local.key'), 
                                     **col_grpc_client_config)

    collaborator = Collaborator(collaborator_id=collaborator_id,
                                wrapped_model=wrapped_model, 
                                channel=channel, 
                                **col_config)


    collaborator.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', '-p', type=str, required=True)
    parser.add_argument('--collaborator_id', '-col', type=str, required=True)
    parser.add_argument('--data_config_fname', '-dc', type=str, default="data.yaml")
    parser.add_argument('--logging_config_fname', '-lc', type=str, default="logging.yaml")
    parser.add_argument('--logging_default_level', '-l', type=str, default="info")
    args = parser.parse_args()
    main(**vars(args))
