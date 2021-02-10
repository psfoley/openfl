# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""DataHandler package."""

from warnings import catch_warnings, simplefilter
import pkgutil

from .data_handler import DataHandler  # NOQA
if pkgutil.find_loader('torch'):
    from .pytorch_data_loader_handler import PyTorchDataLoaderHandler  # NOQA
if pkgutil.find_loader('tensorflow'):
    from .tensorflow_dataset_handler import TensorflowDatasetHandler  # NOQA
from .numpy_data_loader_handler import NumpyDataLoaderHandler
from .data_handler_factory import DataHandlerFactory  # NOQA

