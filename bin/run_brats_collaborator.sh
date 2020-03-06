#!/bin/bash

col_num=$1

../venv/bin/python3 run_collaborator_from_flplan.py -p=brats17_inst2.yaml -col=col_$col_num -dc=docker_data_config.yaml

