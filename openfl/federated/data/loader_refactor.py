# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""DataLoader module."""


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


    def __getattribute__(self, attr):
        """Track access to wrapped DataLoader."""
        if attr not in ['get_loader_data_size','get_access_count','loader','access_count']:
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
