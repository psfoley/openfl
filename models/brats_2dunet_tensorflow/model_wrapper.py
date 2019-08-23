import os
import numpy as np

from .tensorflow2dunet import TensorFlow2DUNet

def load_BraTS17_insitution(path="/opt/datasets/BraTS17", channels_first=False, **kwargs):
    files = ['imgs_train.npy', 'msks_train.npy', 'imgs_val.npy', 'msks_val.npy']
    
    data = [np.load(os.path.join(path, f), mmap_mode='r') for f in files]
    
    if channels_first:
        data = [np.swapaxes(d, 1, 3) for d in data]
        data = [np.swapaxes(d, 2, 3) for d in data]

    return tuple(data)

def get_model():
    X_train, y_train, X_val, y_val = load_BraTS17_insitution(path="/opt/datasets/BraTS17", channels_first=False)
    model = TensorFlow2DUNet(X_train, y_train, X_val, y_val)
    return model