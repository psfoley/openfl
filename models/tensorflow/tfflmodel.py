# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.


import numpy as np
import tqdm
import tensorflow as tf

from models import FLModel


class TensorFlowFLModel(FLModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.assign_ops = None
        self.placeholders = None

        self.tvar_assign_ops = None
        self.tvar_placeholders = None

        # construct the shape needed for the input features
        self.input_shape = (None,) + self.data.get_feature_shape() 

        # child classes should have __init__ function signature (self, data, kwargs), 
        # and should overwrite at least the following while defining the model

        # tensorflow session
        self.sess = None
        # input featrures to the model
        self.X = None
        # input labels to the model
        self.y = None
        # optimizer train step operation
        self.train_step = None
        # model loss function
        self.loss = None
        # model output tensor
        self.output = None
        # function used to validate the model outputs against labels
        self.validation_metric = None
        # tensorflow trainable variables
        self.tvars = None
        # self.optimizer.variables() once self.optimizer is defined
        self.opt_vars = None
        # self.tvars + self.opt_vars 
        self.fl_vars = None

    def train_epoch(self, batch_size=None, use_tqdm=False):
        """
        Train one epoch.

        Returns
        -------
        dict
            {<metric>: <value>}
        """
        tf.keras.backend.set_learning_phase(True)

        losses = []

        gen = self.data.get_train_loader(batch_size)
        if use_tqdm:
            gen = tqdm(gen, desc="training epoch")

        for X, y in gen:
            losses.append(self.train_batch(X, y))

        return np.mean(losses)

    def train_batch(self, X, y):
        feed_dict = {self.X: X, self.y: y}
        
        # run the train step and return the loss
        _, loss = self.sess.run([self.train_step, self.loss], feed_dict=feed_dict)
        return loss

    def validate(self, batch_size=None, use_tqdm=False):
        """
        Run validation.

        Returns
        -------
        dict
            {<metric>: <value>}
        """
        tf.keras.backend.set_learning_phase(False)

        score = 0

        gen = self.data.get_val_loader(batch_size)
        if use_tqdm:
            gen = tqdm(gen, desc="validating")

        for X, y in gen:
            weight = X.shape[0] / self.data.get_validation_data_size()  
            _, s = self.validate_batch(X, y)
            score += s * weight

        return score

    def validate_batch(self, X, y):
        feed_dict = {self.X: X, self.y: y}

        return self.sess.run([self.output, self.validation_metric], feed_dict=feed_dict)

    def get_tensor_dict(self, with_opt_vars=True):
        """
        Get the weights.
        Parameters
        ----------
        with_opt_vars : bool
            Specify if we also want to get the variables of the optimizer.

        Returns
        -------
        dict
            The weight dictionary {<tensor_name>: <value>}
        """
        if with_opt_vars is True:
            variables =  self.fl_vars
        else:
            variables = self.tvars

        # FIXME: do this in one call?
        return {var.name: val for var, val in zip(variables, self.sess.run(variables))}

    def set_tensor_dict(self, tensor_dict, with_opt_vars):
        """
        Set the model weights with a tensor dictionary: {<tensor_name>: <value>}.
        Parameters
        ----------
        tensor_dict : dict
            The model weights dictionary.
        with_opt_vars : bool
            Specify if we also want to set the variables of the optimizer.

        Returns
        -------
        None
        """
        if with_opt_vars:
            self.assign_ops, self.placeholders = \
                tf_set_tensor_dict(tensor_dict, self.sess, self.fl_vars, self.assign_ops, self.placeholders)
        else:
            self.tvar_assign_ops, self.tvar_placeholders = \
                tf_set_tensor_dict(tensor_dict, self.sess, self.tvars, self.tvar_assign_ops, self.tvar_placeholders)

    def reset_opt_vars(self):
        """Reinitialize the optimizer variables."""
        for v in self.opt_vars:
            v.initializer.run(session=self.sess)

    def initialize_globals(self):
        """
        Initialize all global variables
        ----------

        Returns
        -------
        None
        """
        self.sess.run(tf.global_variables_initializer())




# FIXME: what's a nicer construct than this? ugly interface. Perhaps we get an object with an assumed interface that lets is set/get these?
# Note that this will return the assign_ops and placeholder nodes it uses
# if called with None, it will create them.
# to avoid inflating the graph, caller should keep these and pass them back
# What if we want to set a different group of vars in the middle? It is good if it is the subset of the original variables.
def tf_set_tensor_dict(tensor_dict, session, variables, assign_ops=None, placeholders=None):
    if placeholders is None:
        placeholders = {v.name: tf.placeholder(v.dtype, shape=v.shape) for v in variables}
    if assign_ops is None:
        assign_ops = {v.name: tf.assign(v, placeholders[v.name]) for v in variables}

    for k, v in tensor_dict.items():
        session.run(assign_ops[k], feed_dict={placeholders[k]:v})
    
    return assign_ops, placeholders

