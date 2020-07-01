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

    # create the hash of these files
    flplan_fname = os.path.splitext(os.path.basename(plan_path))[0]
    flplan_hash = hash_files(plan_files)

    fed_id = '{}_{}'.format(flplan_fname, flplan_hash[:8])
    agg_id = 'aggregator_{}'.format(fed_id)

    flplan['aggregator']['agg_id'] = agg_id
    flplan['aggregator']['fed_id'] = fed_id
    flplan['collaborator']['agg_id'] = agg_id
    flplan['collaborator']['fed_id'] = fed_id

    return flplan


def hash_files(paths):
    md5 = hashlib.md5()
    for p in paths:
        with open(p, 'rb') as f:
            md5.update(f.read())
    return md5.hexdigest()
