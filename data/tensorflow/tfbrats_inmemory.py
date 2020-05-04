from data import load_from_NIfTY 
from data.tensorflow.tffldata_inmemory import TensorFlowFLDataInMemory


class TensorFlowBratsInMemory(TensorFlowFLDataInMemory):

    def __init__(self, data_path, batch_size, percent_train=0.8, pre_split_shuffle=True, **kwargs):

        super().__init__(batch_size)
        
        X_train, y_train, X_val, y_val = load_from_NIfTY(parent_dir=data_path, 
                                                         percent_train=percent_train, 
                                                         shuffle=pre_split_shuffle, 
                                                         **kwargs)
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
