import os
import numpy as np

from .tensorflow2dunet import TensorFlow2DUNet

def get_model(data=None, data_kwargs={}, model_kwargs={}):
    return TensorFlow2DUNet(data, **model_kwargs)
