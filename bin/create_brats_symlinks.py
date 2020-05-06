#!/usr/bin/env python3

# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.


import argparse
import os
import json


def main(brats_base, symlink_base):
    
    script_dir = os.path.dirname(os.path.realpath(__file__))

    with open(os.path.join(script_dir, 'brain_to_inst.json'), 'r') as f:
        brain_to_inst = json.load(f)

    if not os.path.isdir(brats_base):
        raise FileNotFoundError("brats_base " + brats_base + " not found!")
    if not os.path.isdir(symlink_base):
        raise FileNotFoundError("symlink_base " + symlink_base + " not found!")
    for brain_dir, inst in brain_to_inst.items():
        inst_dir = os.path.join(symlink_base, str(inst))
        if not os.path.isdir(inst_dir):
            os.makedirs(inst_dir)
        src = os.path.join(brats_base, brain_dir)
        dst = os.path.join(inst_dir, brain_dir)
        if os.path.exists(dst):
            raise FileExistsError("dst " + dst + " already exists! Aborting.")
        if not os.path.isdir(src):
            raise FileNotFoundError("Brain dir " + src + " not found!")
        os.symlink(src, dst, target_is_directory=True)
    
    print("Be sure when removing links that you don't remove the target!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--brats_base', '-b', type=str, required=True)
    parser.add_argument('--symlink_base', '-s', type=str, required=True)
    args = parser.parse_args()
    main(**vars(args))
