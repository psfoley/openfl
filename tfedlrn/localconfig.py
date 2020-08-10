# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

import os
import logging
import hashlib
import yaml

from . import load_yaml, get_object

def get_data_path_from_local_config(local_config, collaborator_common_name, data_name_in_local_config):
    data_names_to_paths = local_config['collaborators']

    if collaborator_common_name not in data_names_to_paths:
        raise ValueError("Could not find collaborator id \"{}\" in the local data config file.".format(collaborator_common_name))
    
    data_names_to_paths = data_names_to_paths[collaborator_common_name]
    if data_name_in_local_config not in data_names_to_paths:
        raise ValueError("Could not find data path for collaborator id \"{}\" and dataset name \"{}\".".format(collaborator_common_name, data_name_in_local_config))

    return data_names_to_paths[data_name_in_local_config]
