# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.


import subprocess
import os

def get_available_nvidia_gpus(min_percent_used=0):
    """Determines which Nvidia GPUs are available

    Args:
        float: min_percent_used: Find only GPUs with usage less than or equal to this value

    Returns:
        A list of strings with the GPUs that are available

    """
    gpus = subprocess.run(['nvidia-smi', '--query-gpu=utilization.memory', '--format=csv'], stdout=subprocess.PIPE).stdout.decode('utf-8').split('\n')[1:]
    gpus = [g.split(' %')[0] for g in gpus if len(g) > 0]
    gpus = [int(g) <= min_percent_used for g in gpus]
    gpus = [i for i, g in enumerate(gpus) if g]
    return gpus

def set_cuda_vis_device(gpu):
    """Set which GPUs are visible to the code

    Args:
        gpu: List of GPUs that are visible to the code

    Returns:
        None

    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

def pick_cuda_device(**kwargs):
    """Picks a GPU

    Args:
        **kwargs: Variable parameters to send

    Returns:
        None

    """
    try:
        gpu = get_available_nvidia_gpus(**kwargs)[0]
        set_cuda_vis_device(gpu)
    except FileNotFoundError:
        print("No GPU chosen, nvidia-smi not found")
        pass
