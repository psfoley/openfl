#!/usr/bin/env python3

# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

import argparse
import sys
import os

import yaml

from tfedlrn import parse_fl_plan


def main(plan, out_filepath):

    # FIXME: consistent filesystem (#15)
    script_dir = os.path.dirname(os.path.realpath(__file__))
    base_dir = os.path.join(script_dir, 'federations')
    plan_dir = os.path.join(base_dir, 'plans')

    flplan = yaml.dump(parse_fl_plan(os.path.join(plan_dir, plan)))

    print(flplan)
    if out_filepath is not None:
        with open(out_filepath, 'w') as f:
            f.write(flplan)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', '-p', type=str, required=True)
    parser.add_argument('--out_filepath', '-o', type=str, default=None)
    args = parser.parse_args()
    main(**vars(args))
