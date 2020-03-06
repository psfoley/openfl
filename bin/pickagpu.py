import re
import subprocess
import numpy as np
import os

def get_gpu_mem_util(gpu):
    r = re.compile('([0-9]+)MiB /.+MiB[ |]+([0-9]+)%')
    s = r.search(gpu)
    return int(s.group(1)) + 1, int(s.group(2)) + 1
    
def pick_a_gpu(log_func=print):
    gpus = subprocess.check_output(['nvidia-smi']).decode().split('TITAN Xp')[1:]
    gpus = [get_gpu_mem_util(gpu) for gpu in gpus]
    gpus = [(np.prod(gpu) + np.sum(gpu), i) for i, gpu in enumerate(gpus)]
    gpu = sorted(gpus)[0][1]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log_func('Using GPU {}'.format(os.environ["CUDA_VISIBLE_DEVICES"]))