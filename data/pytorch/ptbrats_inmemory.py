from data import load_from_NIfTY 
from data.pytorch.ptfldata_inmemory import PTFLDataInMemory





class PTBratsInMemory(PTFLDataInMemory):

    def __init__(self, data_path, batch_size, percent_train=0.8, pre_split_shuffle=True, **kwargs):

        super().__init__(batch_size)
        
        X_train, y_train, X_val, y_val = load_from_NIfTY(parent_dir=data_path, 
                                                        percent_train=percent_train, 
                                                        shuffle=pre_split_shuffle, 
                                                        **kwargs)
        self.train_loader = self.create_loader(self, X=X_train, y=y_train, batch_size=self.batch_size)
        self.val_loader = self.create_loader(self, X=X_val, y=y_val, batch_size=self.batch_size)

        


