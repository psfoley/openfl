import os
import numpy as np

from .pytorch2dunet import PyTorch2DUNet

def get_model(data=None, data_kwargs={}, model_kwargs={}):
    if data == None:
        data = BratsData(**data_kwargs)
    return PyTorch2DUNet(data, **model_kwargs)
