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


def main():
    num_collaborators = 1
    agg_proc = execute_process({'exe': './simple_pytorch_mnist_agg.py', 'num_collaborators':num_collaborators})
    col_procs = [execute_process({'exe': './simple_pytorch_mnist_col.py', 
                                  'num_collaborators':num_collaborators,
                                  'col_num':i}, cuda_num=i)
                for i in range(num_collaborators)]

    while agg_proc.poll() is None:
        time.sleep(1)

    for p in col_procs:
        while p.poll() is None:
            time.sleep(1)


if __name__ == '__main__':    
    import argparse
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(**vars(args))
