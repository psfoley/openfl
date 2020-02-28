#!/usr/bin/env python3
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

    # FIXME: this should be easily consumed by the Aggregator object (#16)
    agg_config = flplan['aggregator']
    col_ids = agg_config['collaborators']
    agg = Aggregator(agg_config['id'],
                     flplan['federation'],
                     col_ids,
                     os.path.join(weights_dir, agg_config['initial_weights']),
                     os.path.join(weights_dir, agg_config['latest_weights']),
                     os.path.join(weights_dir, agg_config['best_weights']))

    tls_config = flplan['tls']
    cert_dir = os.path.join(base_dir, 'certs', tls_config['cert_folder'])

    agg_grpc_server = AggregatorGRPCServer(agg)
    agg_grpc_server.serve('[::]',
                          agg_config['port'],
                          disable_tls=tls_config['disable'],
                          disable_client_auth=tls_config['disable_client_auth'],
                          ca=os.path.join(cert_dir, 'ca.crt'),
                          certificate=os.path.join(cert_dir, 'local.crt'),
                          private_key=os.path.join(cert_dir, 'local.key'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', '-p', type=str, required=True)
    parser.add_argument('--logging_config_path', '-c', type=str, default="logging.yaml")
    parser.add_argument('--logging_default_level', '-l', type=str, default="info")
    args = parser.parse_args()
    main(**vars(args))
