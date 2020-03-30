#!/bin/bash

# Copyright (C) 2020 Intel Corporation
# Licensed subject to Collaboration Agreement dated February 28th, 2020 between Intel Corporation and Trustees of the University of Pennsylvania.

col_num=$1

python3 run_collaborator_from_flplan.py -p tf_2dunet_brats_insts2_3.yaml -col col_$col_num -dc docker_data_config.yaml

