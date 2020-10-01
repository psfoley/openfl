# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

from warnings import catch_warnings, simplefilter
    
with catch_warnings():
    simplefilter(action = 'ignore')
    import tensorflow # ignore deprecation warnings in command-line interface


from .runner       import             TaskRunner
from .runner_tf    import   TensorFlowTaskRunner
from .runner_pt    import      PyTorchTaskRunner
from .runner_keras import        KerasTaskRunner
