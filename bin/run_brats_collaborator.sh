#!/bin/bash

col_id=$1

../venv/bin/python3 run_collaborator_from_flplan.py -p=brats17_a.yaml -col=col_$col_id -dc=docker_data.yaml

