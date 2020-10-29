# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

import logging
import copy
import numpy as np
import tensorflow.compat.v1        as tf
tf.disable_v2_behavior()
import tensorflow.compat.v1.keras  as keras
from tensorflow.compat.v1.keras import backend as K
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from .runner_keras import KerasTaskRunner

class FederatedModel(KerasTaskRunner):
    """A wrapper for all task runners

    """
    def __init__(self, build_model, **kwargs):
        """Initializer

        Args:
            model:    build_model function
            **kwargs: Additional parameters to pass to the function
            
        """
        super().__init__(**kwargs)

        self.build_model = build_model

        self.model = self.build_model(self.feature_shape,self.data_loader.num_classes)

        #print(self.model.summary())
        #if self.data is not None:
        #print("Training set size: %d; Validation set size: %d" % (self.get_training_data_size(), self.get_validation_data_size()))
        # initialize the optimizer variables
        
        opt_vars = self.model.optimizer.variables()

        for v in opt_vars:
            v.initializer.run(session = self.sess)

        self.initialize_tensorkeys_for_functions()

    def setup(self,num_collaborators):
        """Create new models for all of the collaborators in the experiment

        Args:
            num_collaborators:  Number of experiment collaborators

        Returns:
            List of models"""
        return [FederatedModel(self.build_model,data_loader=data_slice) for data_slice in self.data_loader.split(num_collaborators,equally=True)]


    def save(self,model_name):
        """
        Save the model in its native format. Because keras is supported today, just call model.save
        """
        self.model.save(model_name)

    def __hash__(self):
        """Return a hash of the model structure"""
        return md5.model.summary()


