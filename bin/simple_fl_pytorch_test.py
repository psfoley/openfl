#!/usr/bin/env python3
import subprocess
import time
import os


def execute_process(args, cuda_num=None):
    array = [args['exe']]
    for k, v in args.items():
        if k == 'exe':
            continue
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


def main(num_collaborators, model_id, device, cuda_device_list):
    if cuda_device_list == -1:
        cuda_device_list = list(range(num_collaborators))
    agg_proc = execute_process({'exe': './simple_fl_agg.py',
                                'num_collaborators':num_collaborators,
                                'initial_model':model_id,
                               })
    col_procs = [execute_process({'exe': './simple_fl_pytorch_col.py', 
                                  'num_collaborators':num_collaborators,
                                  'col_num':i,
                                  'model_id':model_id,
                                  'device':device},
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
    parser.add_argument('--model_id', '-m', type=str, choices=['PyTorchMNISTCNN', 'PyTorch2DUNet'], required=True)
    parser.add_argument('--device', '-d', type=str, default='cuda')
    parser.add_argument('--cuda_device_list', '-c', nargs='+', type=int, default=-1)
    args = parser.parse_args()
    main(**vars(args))
