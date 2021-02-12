from openfl.federated.data.handlers import DataHandler

class PyTorchDataLoaderHandler(DataHandler):

    def __init__(self):
        pass

    @staticmethod
    def get_dependencies():
        return ['torch']

    @staticmethod
    def type():
        from torch.utils.data import DataLoader
        return DataLoader

    def shard_data(self,loader,rank,_size):
        import torch
        from torch.utils.data import DataLoader
        slice_idx = list(range(rank, len(loader.dataset), _size))
        subset = torch.utils.data.Subset(loader.dataset, slice_idx)
        return DataLoader(dataset=subset,batch_size=loader.batch_size,
            num_workers=loader.num_workers,generator=loader.generator)
