from torch.utils.data import DataLoader
import torch
from .data_handler import DataHandler

class PyTorchDataLoaderHandler(DataHandler):

    def __init__(self):
        pass

    def shard_data(self,loader,rank,_size):
        slice_idx = list(range(rank, len(loader.dataset), _size))
        subset = torch.utils.data.Subset(loader.dataset, slice_idx)
        return DataLoader(dataset=subset,batch_size=loader.batch_size,
            num_workers=loader.num_workers,generator=loader.generator)
