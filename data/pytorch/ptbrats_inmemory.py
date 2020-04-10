from data import load_from_NIfTY 
from data.pytorch.ptfldata_inmemory import PyTorchFLDataInMemory





class PyTorchBratsInMemory(PyTorchFLDataInMemory):

    def __init__(self, 
                 data_path, 
                 batch_size, 
                 percent_train=0.8, 
                 pre_split_shuffle=True, 
                 channels_last=False,
                 **kwargs):

        super().__init__(batch_size)
        
        X_train, y_train, X_val, y_val = load_from_NIfTY(parent_dir=data_path, 
                                                        percent_train=percent_train, 
                                                        shuffle=pre_split_shuffle, 
                                                        channels_last=channels_last, 
                                                        **kwargs)
        self.train_loader = self.create_loader(X=X_train, y=y_train)
        self.val_loader = self.create_loader(X=X_val, y=y_val)

        

