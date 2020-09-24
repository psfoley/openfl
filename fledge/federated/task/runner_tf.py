# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

import tensorflow as tf

from .runner_tf_1 import TensorFlowV1TaskRunner
from .runner_tf_2 import TensorFlowV2TaskRunner
from .runner      import TaskRunner

#Check if the installed version of Tensorflow is 1.x
tf1 = int(tf.__version__[0]) == 1

class TensorFlowTaskRunner(TensorFlowV1TaskRunner if tf1 else TensorFlowV2TaskRunner):
    """
    Wrapper for compatibility with V1 and V2 models in the Federated Learning solution
    """
    def __init__(self, **kwargs):
        """
        Initializer

        Args:
            **kwargs: Additional parameters to pass to the function
        """
        
        super().__init__(**kwargs)
