# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from openfl.federated.data.handlers import DataHandlerFactory
from logging import getLogger

"""DataLoader module."""

logger = getLogger(__name__)


class FederatedDataLoader(object):
    """Federated Learning Data Loader Class."""

    def __init__(self, loader, **kwargs):
        """
        Instantiate the data object.

        Returns:
            None
        """
        self.loader = loader
        self.access_count = 0
        self.shard_defined = False

    def __getattribute__(self, attr):
        """Track access to wrapped DataLoader."""
        if attr not in ['get_loader_data_size','get_access_count','loader','access_count','shard_data']:
            logger.debug(f'{attr} accessed')
            if attr is not '__class__':
                self.access_count += 1
            return self.loader.__getattribute__(attr)
        return super(FederatedDataLoader, self).__getattribute__(attr)

    def __iter__(self):
        return self.loader.__iter__()

    def __len__(self):
        return len(self.loader)


    def get_loader_data_size(self):

        """
        Get total number of samples associated with DataLoader.

        Returns:
            int: number of samples
        """
        try:
            length = len(self.loader)
        except Exception:
            length = 1
            pass

        return length

    def get_access_count(self):
        """Return the number of times the DataLoader has been accessed"""
        return self.access_count

    def shard_data(self,rank=0,federation_size=1):
        """Reinitialize dataloader with correct shard"""
        loader_type_factory = DataHandlerFactory()
        if loader_type_factory.is_supported(self.loader):
            data_loader_type_handler = loader_type_factory.get_data_handler(self.loader)
            self.loader = data_loader_type_handler.shard_data(self.loader,rank,federation_size)
        else:
            logger.info(f'Data of type {type(self.loader)} does not have a data type handler defined') 

        self.shard_defined = True
