#!/usr/bin/env python3

# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

import argparse
import os
import logging

from tfedlrn.aggregator.aggregator_rework_strawman import Aggregator
from tfedlrn.comms.grpc.aggregatorgrpcserver import AggregatorGRPCServer
from tfedlrn import parse_fl_plan, load_yaml, get_object
from tfedlrn.tensor_transformation_pipelines import get_compression_pipeline 
from tfedlrn import TensorKey

from setup_logging import setup_logging


def main(plan, collaborators_file, single_col_cert_common_name, logging_config_path, logging_default_level):
    setup_logging(path=logging_config_path, default_level=logging_default_level)

    # FIXME: consistent filesystem (#15)
    script_dir = os.path.dirname(os.path.realpath(__file__))
    base_dir = os.path.join(script_dir, 'federations')
    plan_dir = os.path.join(base_dir, 'plans')
    weights_dir = os.path.join(base_dir, 'weights')
    collaborators_dir = os.path.join(base_dir, 'collaborator_lists')

    flplan = parse_fl_plan(os.path.join(plan_dir, plan))
    agg_config = flplan['aggregator']
    network_config = flplan['network']
    tasks = flplan['tasks']
    task_assigner_config = flplan['task_assigner']

    # patch in the collaborators file
    agg_config['collaborator_common_names'] = load_yaml(os.path.join(collaborators_dir, collaborators_file))['collaborator_common_names']

    init_model_fpath = os.path.join(weights_dir, agg_config['init_model_fname'])

    if flplan.get('compression_pipeline') is not None:
        compression_pipeline = get_compression_pipeline(**flplan.get('compression_pipeline'))
    else:
        compression_pipeline = None

    #Create task assigner
    task_assigner = get_object(**task_assigner_config,
                               tasks=tasks,
                               collaborator_list=agg_config['collaborator_common_names'],
                               rounds=agg_config['rounds_to_train'])

    #custom_tensor_dir has any other custom tensors that the user wants present on the aggregator on initialization 
    agg = Aggregator(initial_model_file_path=init_model_fpath,
                     custom_tensor_dir=None,
                     compression_pipeline=compression_pipeline,
                     task_assigner=task_assigner,
                     single_col_cert_common_name=single_col_cert_common_name,
                     **agg_config)

    # default cert folder to pki
    cert_dir = os.path.join(base_dir, network_config.pop('cert_folder', 'pki')) # default to 'pki'

    cert_common_name = network_config['agg_addr']
    agg_cert_path = os.path.join(cert_dir, "agg_{}".format(cert_common_name))

    agg_grpc_server = AggregatorGRPCServer(agg)
    agg_grpc_server.serve(ca=os.path.join(cert_dir, 'cert_chain.crt'),
                          certificate=os.path.join(agg_cert_path, 'agg_{}.crt'.format(cert_common_name)),
                          private_key=os.path.join(agg_cert_path, 'agg_{}.key'.format(cert_common_name)), 
                          **network_config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', '-p', type=str, required=True)
    parser.add_argument('--collaborators_file', '-c', type=str, required=True, help="Name of YAML File in /bin/federations/collaborator_lists/")
    parser.add_argument('--single_col_cert_common_name', '-scn', type=str, default=None)
    parser.add_argument('--logging_config_path', '-lcp', type=str, default="logging.yaml")
    parser.add_argument('--logging_default_level', '-l', type=str, default="info")
    args = parser.parse_args()
    main(**vars(args))
