# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

import os
import logging
import hashlib
import yaml

from . import load_yaml


def parse_fl_plan(plan_path, logger=None):
    if logger is None:
        logger = logging.getLogger(__name__)

    flplan = load_yaml(plan_path)

    # collect all the plan filepaths used
    plan_files = [plan_path]

    # walk the top level keys for defaults_file in sorted order
    for k in sorted(flplan.keys()):
        defaults_file = flplan[k].get('defaults_file')
        if defaults_file is not None:
            defaults_file = os.path.join(os.path.dirname(plan_path), defaults_file)
            logger.info("Using FLPlan defaults for section '{}' from file '{}'".format(k, defaults_file))
            defaults = load_yaml(defaults_file)
            defaults.update(flplan[k])
            flplan[k] = defaults
            plan_files.append(defaults_file)

    # create the hash of these files
    flplan_fname = os.path.splitext(os.path.basename(plan_path))[0]
    flplan_hash = hash_files(plan_files, logger=logger)

    # check for auto_port convenience settings
    if flplan['network']['agg_port'] == 'auto':
        # replace the port number with something in the range of min-max
        # default is 49152 to 60999
        port_range = flplan['network'].get('agg_auto_port_range', (49152, 60999))
        port = (int(flplan_hash, 16) % (port_range[1] - port_range[0])) + port_range[0]
        flplan['network']['agg_port'] = port

    federation_uuid = '{}_{}'.format(flplan_fname, flplan_hash[:8])
    aggregator_uuid = 'aggregator_{}'.format(federation_uuid)

    flplan['aggregator']['aggregator_uuid'] = aggregator_uuid
    flplan['aggregator']['federation_uuid'] = federation_uuid
    flplan['collaborator']['aggregator_uuid'] = aggregator_uuid
    flplan['collaborator']['federation_uuid'] = federation_uuid

    logger.info("Parsed plan:\n{}".format(yaml.dump(flplan)))

    return flplan


def hash_files(paths, logger=None):
    md5 = hashlib.md5()
    for p in paths:
        with open(p, 'rb') as f:
            md5.update(f.read())
        if logger is not None:
            logger.info("After hashing {}, hash is {}".format(p, md5.hexdigest()))
    return md5.hexdigest()
