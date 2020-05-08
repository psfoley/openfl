#!/bin/bash

# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

col_num=$1

python3 run_collaborator_from_flplan.py -p tf_2dunet_brats_insts2_3.yaml -col col_$col_num -dc docker_data_config.yaml

