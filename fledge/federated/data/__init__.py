# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed
# evaluation license agreement between Intel Corporation and you.

from warnings import catch_warnings, simplefilter
import pkgutil

with catch_warnings():
    simplefilter(action='ignore', category=FutureWarning)
    if pkgutil.find_loader('tensorflow'):
        # ignore deprecation warnings in command-line interface
        import tensorflow  # NOQA

from .loader import DataLoader  # NOQA

if pkgutil.find_loader('tensorflow'):
    from .loader_tf import TensorFlowDataLoader  # NOQA
    from .loader_keras import KerasDataLoader  # NOQA
    from .federated_data import FederatedDataSet  # NOQA

if pkgutil.find_loader('torch'):
    from .loader_pt import PyTorchDataLoader  # NOQA
    from .federated_data import FederatedDataSet  # NOQA
if pkgutil.find_loader('torch') and pkgutil.find_loader('tensorflow'):
    from .loader_fe import FastEstimatorDataLoader  # NOQA
