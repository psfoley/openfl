#!/usr/bin/env python3
import subprocess
import time
import os
import sys

import tfedlrn
from tfedlrn.gpuutils import get_available_nvidia_gpus

def execute_process(*args, cuda_num=None, **kwargs):
    array = list(args)
    for k, v in kwargs.items():
        array.append('--{}'.format(k))
        if isinstance(v, list):
            for t in v:
                array.append(str(t))
        else:
            array.append(str(v))

    env = os.environ.copy()
    if cuda_num is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_num)

    return subprocess.Popen(array, env=env)


def main(num_collaborators, model_id):
    script_dir = os.path.dirname(os.path.realpath(__file__))

    cuda_device_list = get_available_nvidia_gpus()
    if len(cuda_device_list) == 0:
        raise RuntimeError("No available GPUs for collaborators!")

    if num_collaborators <= 0:
        num_collaborators = len(cuda_device_list)
        print("*** SETTING NUM COLLABORATORS TO {}, BASED ON AVAILABLE GPUS".format(num_collaborators))

    if len(cuda_device_list) < num_collaborators:
        raise RuntimeError("Not enough available GPUs for each collaborator! Available GPUs: {} Num Collaborators: {}".format(len(cuda_device_list), num_collaborators))

    agg_proc = execute_process(sys.executable,
                               '{}/simple_fl_agg.py'.format(script_dir),
                               num_collaborators=num_collaborators,
                               initial_model=model_id
                               )
    col_procs = [execute_process(sys.executable,
                                 '{}/simple_fl_tensorflow_col.py'.format(script_dir),
                                 num_collaborators=num_collaborators,
                                 col_num=i,
                                 model_id=model_id,
                                 cuda_num=cuda_device_list[i])
                for i in range(num_collaborators)]

    while agg_proc.poll() is None:
        time.sleep(1)

    for p in col_procs:
        while p.poll() is None:
            time.sleep(1)


if __name__ == '__main__':    
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_collaborators', '-n', type=int, default=4)
    parser.add_argument('--model_id', '-m', type=str, choices=['TensorFlow2DUNet'], required=True)
    args = parser.parse_args()
    main(**vars(args))
