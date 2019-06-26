import numpy as np


def shuffle_data(X, y, seed=None):
    if seed is not None:
        np.random.seed(seed)
    idx = np.random.permutation(np.arange(X.shape[0]))
    return X[idx], y[idx]


class FLModel(object):

    def __init__(self):
    	