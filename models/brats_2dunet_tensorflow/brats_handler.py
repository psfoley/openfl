import numpy as np
from .datasets import load_from_NIfTY 



class BratsHandler(object):

    def __init__(self, dataset_path, percent_train, shuffle, **kwargs):

        X_train, y_train, X_val, y_val = load_from_NIfTY(parent_dir=dataset_path, 
                                                        percent_train=percent_train, 
                                                        shuffle=shuffle, 
                                                        **kwargs)
        self.data = X_train, y_train, X_val, y_val
        self.feature_shape = X_train[0].shape


