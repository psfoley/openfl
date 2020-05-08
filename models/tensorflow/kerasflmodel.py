# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

"""Base classes for developing a keras.Model() Federated Learning model.

You may copy this file as the starting point of your own keras model.
"""
import logging
import numpy as np
import tensorflow as tf

from models import FLModel

import tensorflow.keras as keras
from tensorflow.keras import backend as K

class KerasFLModel(FLModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.model = keras.Model()

        NUM_PARALLEL_EXEC_UNITS = 1
        config = tf.ConfigProto(intra_op_parallelism_threads=NUM_PARALLEL_EXEC_UNITS, 
                                inter_op_parallelism_threads=1, 
                                allow_soft_placement=True, 
                                device_count = {'CPU': NUM_PARALLEL_EXEC_UNITS })
        config.gpu_options.allow_growth=True
        
        self.sess = tf.Session(config=config)
        K.set_session(self.sess)

        # child class should have an __init__ function with signature: (self, data, **kwargs)
        # and should overwrite self.model with a child of keras.Model

    def train_epoch(self):
        history = self.model.fit(self.data.X_train, 
                                 self.data.y_train,
                                 batch_size=self.data.batch_size,
                                 epochs=1,
                                 verbose=0,)

        # As we are training for one epoch, we only need the first element in each list.
        ret_dict = {name:values[0] for name, values in history.history.items()}
        return ret_dict['loss']

    def validate(self):
        vals = self.model.evaluate(self.data.X_val, self.data.y_val, verbose=0)
        metrics_names = self.model.metrics_names
        ret_dict = dict(zip(metrics_names, vals))
        return ret_dict['acc']

    @staticmethod
    def _get_weights_dict(obj):
        """
        Get the dictionary of weights.
        Parameters
        ----------
        obj : Model or Optimizer
            The target object that we want to get the weights.

        Returns
        -------
        dict
            The weight dictionary.
        """
        weights_dict = {}
        weight_names = [weight.name for weight in obj.weights]
        weight_values = obj.get_weights()
        for name, value in zip(weight_names, weight_values):
            weights_dict[name] = value
        return weights_dict

    @staticmethod
    def _set_weights_dict(obj, weights_dict):
        """
        Set the object weights with a dictionary. The obj can be a model or an optimizer.
        Parameters
        ----------
        obj : Model or Optimizer
            The target object that we want to set the weights.
        weights_dict : dict
            The weight dictionary.

        Returns
        -------
        None
        """
        weight_names = [weight.name for weight in obj.weights]
        weight_values = [weights_dict[name] for name in weight_names]
        obj.set_weights(weight_values)

    def initialize_globals(self):
        self.sess.run(tf.global_variables_initializer())

    def get_tensor_dict(self, with_opt_vars):
        """
        Get the model weights as a tensor dictionary.

        Parameters
        ----------
        with_opt_vars : bool
            If we should include the optimizer's status.

        Returns
        -------
        dict
            The tensor dictionary.
        """
        model_weights = self._get_weights_dict(self.model)

        if with_opt_vars:
            opt_weights = self._get_weights_dict(self.model.optimizer)

            model_weights.update(opt_weights)
            if len(opt_weights) == 0:
                self.logger.debug("WARNING: We didn't find variables for the optimizer.")
        return model_weights

    def set_tensor_dict(self, tensor_dict, with_opt_vars):
        
        if with_opt_vars is False:
            self._set_weights_dict(self.model, tensor_dict)
        else:
            model_weight_names = [weight.name for weight in self.model.weights]
            model_weights_dict = {name: tensor_dict[name] for name in model_weight_names}
            opt_weight_names = [weight.name for weight in self.model.optimizer.weights]
            opt_weights_dict = {name: tensor_dict[name] for name in opt_weight_names}
            self._set_weights_dict(self.model, model_weights_dict)
            self._set_weights_dict(self.model.optimizer, opt_weights_dict)

    def reset_opt_vars(self):
        for weight in self.model.optimizer.weights:
            weight.initializer.run(session=self.sess)
