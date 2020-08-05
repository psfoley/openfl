#!/usr/bin/env python3

# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

import argparse
import sys
import os
import logging
import importlib

from tfedlrn.collaborator.collaborator import Collaborator
from tfedlrn.flplan import create_collaborator_object_from_flplan, parse_fl_plan, load_yaml
from setup_logging import setup_logging


def main(plan, collaborator_common_name, single_col_cert_common_name, data_config_fname, logging_config_path, logging_default_level, logging_directory):
    """Runs the collaborator client process from the federation (FL) plan

    Args:
        plan: The filename for the federation (FL) plan YAML file
        collaborator_common_name: The common name for the collaborator node
        single_col_cert_common_name: The SSL certificate for this collaborator
        data_config_fname: The dataset configuration filename (YAML)
        logging_config_fname: The log file
        logging_default_level: The log level

    """
    # FIXME: consistent filesystem (#15)
    script_dir = os.path.dirname(os.path.realpath(__file__))
    base_dir = os.path.join(script_dir, 'federations')
    plan_dir = os.path.join(base_dir, 'plans')
    logging_config_path = os.path.join(script_dir, logging_config_path)
    logging_directory = os.path.join(script_dir, logging_directory)

    setup_logging(path=logging_config_path, default_level=logging_default_level, logging_directory=logging_directory)

    flplan = parse_fl_plan(os.path.join(plan_dir, plan))

    local_config = load_yaml(os.path.join(base_dir, data_config_fname))

    collaborator = create_collaborator_object_from_flplan(flplan, collaborator_common_name, local_config, base_dir, single_col_cert_common_name)

    collaborator.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', '-p', type=str, required=True)
    parser.add_argument('--collaborator_common_name', '-col', type=str, required=True)
    parser.add_argument('--single_col_cert_common_name', '-scn', type=str, default=None)
    parser.add_argument('--data_config_fname', '-dc', type=str, default="local_data_config.yaml")
    parser.add_argument('--logging_config_path', '-lcp', type=str, default="logging.yaml")
    parser.add_argument('--logging_default_level', '-l', type=str, default="info")
    parser.add_argument('--logging_directory', '-ld', type=str, default="logs")
    args = parser.parse_args()
    main(**vars(args))
