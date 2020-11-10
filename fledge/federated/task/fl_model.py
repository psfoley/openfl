# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

import logging
import copy
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from .runner_keras import KerasTaskRunner
from .runner_pt import PyTorchTaskRunner
from .runner import TaskRunner
from torch import nn

class FederatedModel(TaskRunner):
    """A wrapper for all task runners

    """
    def __init__(self, build_model, optimizer=None, loss_fn=None, **kwargs):
        """Initializer

        Args:
            model:    build_model function
            **kwargs: Additional parameters to pass to the function
            
        """

        super().__init__(**kwargs)

        self.build_model = build_model
        self.lambda_opt = None
        if isinstance(build_model,nn.Module):
            self.model = build_model
            impl = PyTorchTaskRunner
            build_model.__init__()
        else:
            self.model = self.build_model(self.feature_shape,self.data_loader.num_classes)
            impl = KerasTaskRunner

        if optimizer is not None:
            self.optimizer = optimizer(self.model.parameters())
            self.lambda_opt = optimizer
            self.loss_fn = loss_fn
        else:
            self.optimizer = self.model.optimizer
        self.runner = impl(**kwargs)
        self.runner.forward = self.model.forward
        self.runner.model = self.model
        self.runner.optimizer = self.optimizer
        self.runner.loss_fn = self.loss_fn
        self.tensor_dict_split_fn_kwargs = self.runner.tensor_dict_split_fn_kwargs
        self.initialize_tensorkeys_for_functions()

    def __getattribute__(self,attr):
        """
        Direct call into self.runner methods if necessary
        """
        if attr in ['reset_opt_vars','initialize_globals','set_tensor_dict','get_tensor_dict',\
                    'get_required_tensorkeys_for_function','initialize_tensorkeys_for_functions',\
                    'save_native','load_native','rebuild_model','set_optimizer_treatment','train','train_batches','validate']:
            return self.runner.__getattribute__(attr)
        return super(FederatedModel,self).__getattribute__(attr)

    def setup(self,num_collaborators):
        """Create new models for all of the collaborators in the experiment

        Args:
            num_collaborators:  Number of experiment collaborators

        Returns:
            List of models"""
        if self.lambda_opt is not None:
            return [FederatedModel(self.build_model,optimizer=self.lambda_opt,loss_fn=self.loss_fn,data_loader=data_slice) for data_slice in self.data_loader.split(num_collaborators,equally=True)]
        else:
            return [FederatedModel(self.build_model,optimizer=self.optimizer,loss_fn=self.loss_fn,data_loader=data_slice) for data_slice in self.data_loader.split(num_collaborators,equally=True)]


    def save(self,model_name):
        """
        Save the model in its native format. Because keras is supported today, just call model.save
        """
        self.model.save(model_name)

    def __hash__(self):
        """Return a hash of the model structure"""
        return md5.model.summary()


