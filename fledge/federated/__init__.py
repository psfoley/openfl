# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

import pkgutil
from .plan import Plan
from .task import TaskRunner
from .data import DataLoader
if (True if pkgutil.find_loader('tensorflow') else False):
  from .task import TensorFlowTaskRunner, KerasTaskRunner, FederatedModel
  from .data import TensorFlowDataLoader, KerasDataLoader, FederatedDataSet
if (True if pkgutil.find_loader('torch') else False):
  from .task import PyTorchTaskRunner, FederatedModel
  from .data import PyTorchDataLoader, FederatedDataSet
if (True if pkgutil.find_loader('torch') else False) and (True if pkgutil.find_loader('tensorflow') else False):
  from .task import FastEstimatorTaskRunner
  from .data import FastEstimatorDataLoader

