import os
import numpy as np

from .datasets import load_from_NIfTY
from .tensorflow2dunet import TensorFlow2DUNet
from .brats_data import BratsData 

def get_model(data=None, 
             data_path=None, 
             percent_train=0.8, 
             shuffle=True, 
             **kwargs):
    if data == None: 
        data = BratsData(data_path = data_path, 
                         percent_train=percent_train, 
                         shuffle = shuffle, 
                         **kwargs)
    return TensorFlow2DUNet(data)
