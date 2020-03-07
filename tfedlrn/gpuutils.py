import subprocess
import os

def get_available_nvidia_gpus(min_percent_used=0):
    gpus = subprocess.run(['nvidia-smi', '--query-gpu=utilization.memory', '--format=csv'], stdout=subprocess.PIPE).stdout.decode('utf-8').split('\n')[1:]
    gpus = [g.split(' %')[0] for g in gpus if len(g) > 0]
    gpus = [int(g) <= min_percent_used for g in gpus]
    gpus = [i for i, g in enumerate(gpus) if g]
    return gpus

def set_cuda_vis_device(gpu):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

def pick_cuda_device(**kwargs):
    try:
        gpu = get_available_nvidia_gpus(**kwargs)[0]
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    except FileNotFoundError:
        print("No GPU chosen, nvidia-smi not found")
        pass
