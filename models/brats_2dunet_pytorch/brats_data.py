import numpy as np
from .datasets import load_from_NIfTY 



class BratsData(object):

    def __init__(self, data_path, percent_train=0.8, shuffle=True, **kwargs):

        X_train, y_train, X_val, y_val = load_from_NIfTY(parent_dir=data_path, 
                                                        percent_train=percent_train, 
                                                        shuffle=shuffle, 
                                                        **kwargs)
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        
    def get_feature_shape(self):
        return self.X_train[0].shape
    
    def get_training_data(self):
        return self.X_train, self.y_train
    
    def get_validation_data(self):
        return self.X_val, self.y_val

    def get_training_data_size(self):
        return self.X_train.shape[0]

    def get_validation_data_size(self):
        return self.X_val.shape[0]


