import os
import numpy as np

from .tensorflow2dunet import TensorFlow2DUNet
from .brats_data import BratsData 

def get_model(data=None, data_kwargs={}, model_kwargs={}):
    if data == None:
        data = BratsData(**data_kwargs)
    return TensorFlow2DUNet(data, **model_kwargs)
