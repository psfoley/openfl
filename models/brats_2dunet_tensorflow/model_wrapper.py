import os
import numpy as np

from .datasets import load_from_NIfTY
from .tensorflow2dunet import TensorFlow2DUNet

def get_model(dataset_path, **kwargs):
    if dataset_path is not None:
        X_train, y_train, X_val, y_val = load_from_NIfTY(dataset_path, **kwargs)
    else:
        X_train = None
        y_train = None
        X_val = None
        y_val = None
    return TensorFlow2DUNet(X_train, y_train, X_val, y_val)
