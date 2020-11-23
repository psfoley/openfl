# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed
# evaluation license agreement between Intel Corporation and you.

import pkgutil
from .plan import Plan  # NOQA
from .task import TaskRunner  # NOQA
from .data import DataLoader  # NOQA

if pkgutil.find_loader('tensorflow'):
    from .task import TensorFlowTaskRunner, KerasTaskRunner, FederatedModel  # NOQA
    from .data import TensorFlowDataLoader, KerasDataLoader, FederatedDataSet  # NOQA
if pkgutil.find_loader('torch'):
    from .task import PyTorchTaskRunner, FederatedModel  # NOQA
    from .data import PyTorchDataLoader, FederatedDataSet  # NOQA
if pkgutil.find_loader('torch') and pkgutil.find_loader('tensorflow'):
    from .task import FastEstimatorTaskRunner  # NOQA
    from .data import FastEstimatorDataLoader  # NOQA
