# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

from warnings import catch_warnings, simplefilter
import pkgutil
    
with catch_warnings():
    simplefilter(action = 'ignore', category = FutureWarning)
    if (True if pkgutil.find_loader('tensorflow') else False):
      import tensorflow # ignore deprecation warnings in command-line interface

from .loader         import      DataLoader
if (True if pkgutil.find_loader('tensorflow') else False):
  from .loader_tf    import      TensorFlowDataLoader
  from .loader_keras import      KerasDataLoader
  from .federated_data import    FederatedDataSet
if (True if pkgutil.find_loader('torch') else False):
  from .loader_pt    import      PyTorchDataLoader
  from .federated_data import    FederatedDataSet
if (True if pkgutil.find_loader('torch') else False) and (True if pkgutil.find_loader('tensorflow') else False):
  from .loader_fe    import      FastEstimatorDataLoader
