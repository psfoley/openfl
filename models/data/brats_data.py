import numpy as np
from math import ceil
from tqdm import tqdm
from .datasets import load_from_NIfTY 



class BratsData(object):

    def __init__(self, data_path, batch_size, percent_train=0.8, pre_split_shuffle=True, **kwargs):

        X_train, y_train, X_val, y_val = load_from_NIfTY(parent_dir=data_path, 
                                                        percent_train=percent_train, 
                                                        shuffle=pre_split_shuffle, 
                                                        **kwargs)
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

        self.batch_size=64
        
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

    @staticmethod
    def batch_generator(X, y, idxs, batch_size, num_batches):
        for i in range(num_batches):
            a = i * batch_size
            b = a + batch_size
            yield X[idxs[a:b]], y[idxs[a:b]]

    def get_batch_generator(self, dtype, batch_size=None, use_tqdm=False):
        if batch_size == None:
            batch_size = self.batch_size

        if dtype == 'train':
            X = self.X_train
            y = self.y_train
        elif dtype == 'val':
            X = self.X_val
            y = self.y_val
        else:
            raise ValueError('dtype needs to be train or val')

        # shuffle data indices
        idxs = np.random.permutation(np.arange(X.shape[0]))

        # compute the number of batches
        num_batches = ceil(X.shape[0] / batch_size)

        # build the generator
        gen = self.batch_generator(X, y, idxs, batch_size, num_batches)
        if use_tqdm:
            gen = tqdm(gen, desc="training epoch")
        
        return gen
        


