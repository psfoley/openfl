import os
import numpy as np

from .pytorch2dunet import PyTorch2DUNet
from .brats_data import BratsData 

def get_model(data=None, 
             data_path=None, 
             percent_train=0.8, 
             **kwargs):
    if data == None: 
        data = BratsData(data_path=data_path, 
                         percent_train=percent_train, 
                         **kwargs)
    return PyTorch2DUNet(data)
