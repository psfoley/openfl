import os
import numpy as np

from .datasets import load_from_NIfTY
from .tensorflow2dunet import TensorFlow2DUNet
from .brats_handler import BratsHandler 

def get_model(data_handler=None, 
             dataset_path=None, 
             percent_train=None, 
             shuffle=None, 
             **kwargs):
    if data_handler == None: 
        data_handler = BratsHandler(dataset_path, percent_train, shuffle, **kwargs)
    return TensorFlow2DUNet(data_handler)
