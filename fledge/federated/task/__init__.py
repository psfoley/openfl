# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

from warnings import catch_warnings, simplefilter
import pkgutil
    
with catch_warnings():
    simplefilter(action = 'ignore', category = FutureWarning)
    if (True if pkgutil.find_loader('tensorflow') else False):
      import tensorflow # ignore deprecation warnings in command-line interface

from .runner       import           TaskRunner
if (True if pkgutil.find_loader('tensorflow') else False):
  from .runner_tf    import TensorFlowTaskRunner
  from .runner_keras import KerasTaskRunner
  from .fl_model     import FederatedModel
if (True if pkgutil.find_loader('torch') else False):
  from .runner_pt    import         PyTorchTaskRunner
if (True if pkgutil.find_loader('torch') else False) and (True if pkgutil.find_loader('tensorflow') else False):
  from .runner_fe    import         FastEstimatorTaskRunner
