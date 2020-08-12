# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.


import numpy as np
import tqdm
import tensorflow as tf

from models import FLModel
from tfedlrn import TensorKey,split_tensor_dict_for_holdouts


class TensorFlowFLModel(FLModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.assign_ops = None
        self.placeholders = None

        self.tvar_assign_ops = None
        self.tvar_placeholders = None

        # construct the shape needed for the input features
        self.input_shape = (None,) + self.data.get_feature_shape() 
        
        #Required tensorkeys for all public functions in TensorFlowFLModel
        self.required_tensorkeys_for_function = {}

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

    def rebuild_model(self, round, input_tensor_dict):
        """
        Parse tensor names and update weights of model. Handles the optimizer treatment

        Returns
        -------
        None
        """
        if self.opt_treatment == 'RESET':
            self.reset_opt_vars()
            self.set_tensor_dict(input_tensor_dict,with_opt_vars=False)
        elif round > 0 and self.opt_treatment == 'CONTINUE_GLOBAL':
            self.set_tensor_dict(input_tensor_dict,with_opt_vars=True)
        else:
            self.set_tensor_dict(input_tensor_dict,with_opt_vars=False)

    def train_batches(self, col_name, round_num, input_tensor_dict, num_batches, use_tqdm=False,**kwargs):
        """
        Perform the training for a specified number of batches. Is expected to perform draws randomly, without 
        replacement until data is exausted. Then data is replaced and shuffled and draws continue.

        Returns
        -------
        float
            loss
        """
        batch_size = self.data.batch_size

        if kwargs['batch_size']:
            batch_size = kwargs['batch_size']

        #rebuild model with updated weights
        self.rebuild_model(round_num, input_tensor_dict)

        tf.keras.backend.set_learning_phase(True)

        losses = []
        batch_num = 0

        while batch_num < num_batches:
            # get iterator for batch draws (shuffling happens here)
            gen = self.data.get_train_loader(batch_size)
            if use_tqdm:
                gen = tqdm.tqdm(gen, desc="training epoch")

            for (X, y) in gen:
                if batch_num >= num_batches:
                    break
                else: 
                    losses.append(self.train_batch(X, y))
                    batch_num += 1

        #Output metric tensors (scalar)
        origin = col_name
        tags = ('trained',)
        output_metric_dict = {TensorKey(self.loss_name,origin,round_num,('metric',)): np.array(np.mean(losses))}

        #output model tensors (Doesn't include TensorKey)
        output_model_dict = self.get_tensor_dict(with_opt_vars=True)
        global_model_dict,local_model_dict = split_tensor_dict_for_holdouts(self.logger, output_model_dict, **self.tensor_dict_split_fn_kwargs)

        #Create global tensorkeys
        global_tensorkey_model_dict = {TensorKey(tensor_name,origin,round_num,tags): nparray for tensor_name,nparray in global_model_dict.items()}
        #Create tensorkeys that should stay local
        local_tensorkey_model_dict = {TensorKey(tensor_name,origin,round_num,tags): nparray for tensor_name,nparray in local_model_dict.items()}
        #The train/validate aggregated function of the next round will look for the updated model parameters. 
        #This ensures they will be resolved locally
        next_local_tensorkey_model_dict = {TensorKey(tensor_name,origin,round_num+1,('model',)): nparray for tensor_name,nparray in local_model_dict.items()}


        global_tensor_dict = {**output_metric_dict,**global_tensorkey_model_dict}
        local_tensor_dict = {**local_tensorkey_model_dict,**next_local_tensorkey_model_dict}

        #Update the required tensors if they need to be pulled from the aggregator
        #TODO this logic can break if different collaborators have different roles between rounds.
        #For example, if a collaborator only performs validation in the first round but training
        #in the second, it has no way of knowing the optimizer state tensor names to request from the aggregator
        #because these are only created after training occurs. A work around could involve doing a single epoch of training
        #on random data to get the optimizer names, and then throwing away the model.
        if self.opt_treatment == 'CONTINUE_GLOBAL':
            self.initialize_tensorkeys_for_functions()

        #Return global_tensor_dict, local_tensor_dict
        return global_tensor_dict,local_tensor_dict


    def train_batch(self, X, y):
        feed_dict = {self.X: X, self.y: y}
        
        # run the train step and return the loss
        _, loss = self.sess.run([self.train_step, self.loss], feed_dict=feed_dict)
        return loss

    def validate(self, col_name, round_num, input_tensor_dict, use_tqdm=False,**kwargs):
        """
        Run validation.

        Returns
        -------
        dict
            {<metric>: <value>}
        """
        batch_size = self.data.batch_size

        if kwargs['batch_size']:
            batch_size = kwargs['batch_size']

        self.rebuild_model(round_num, input_tensor_dict)
        tf.keras.backend.set_learning_phase(False)

        score = 0

        gen = self.data.get_val_loader(batch_size)
        if use_tqdm:
            gen = tqdm.tqdm(gen, desc="validating")

        for X, y in gen:
            weight = X.shape[0] / self.data.get_validation_data_size()  
            _, s = self.validate_batch(X, y)
            score += s * weight

        origin = col_name
        suffix = 'validate'
        if kwargs['local_model'] == True:
            suffix += '_local'
        else:
            suffix += '_agg'
        tags = ('metric',suffix)
        output_tensor_dict = {TensorKey(self.validation_metric_name,origin,round_num,tags): np.array(score)}

        #Return empty dict for local metrics
        return output_tensor_dict,{}

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

    def _get_weights_names(self, with_opt_vars=True):
        """
        Get the weights.
        Parameters
        ----------
        with_opt_vars : bool
            Specify if we also want to get the variables of the optimizer.

        Returns
        -------
        list
            The weight names list
        """
        if with_opt_vars is True:
            variables =  self.fl_vars
        else:
            variables = self.tvars

        return [var.name for var in variables]

    def get_required_tensorkeys_for_function(self, func_name, **kwargs):
        """
        Get the required tensors for specified function that could be called as part of a task.
        By default, this is just all of the layers and optimizer of the model. 

        Parameters
        ----------
        None

        Returns
        -------
        List
            [TensorKey]
        """

        if func_name == 'validate':
            local_model = 'local_model=' + str(kwargs['local_model'])
            return self.required_tensorkeys_for_function[func_name][local_model]
        else:
            return self.required_tensorkeys_for_function[func_name]


    def initialize_tensorkeys_for_functions(self,with_opt_vars=False):
        """
        Set the required tensors for all publicly accessible methods that could be called as part of a task.
        By default, this is just all of the layers and optimizer of the model. Custom tensors should be added to this function

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        #Minimal required tensors for train function
        tensor_names = self._get_weights_names(with_opt_vars=with_opt_vars)
        self.logger.debug('Initial model tensor names: {}'.format(tensor_names))
        self.required_tensorkeys_for_function['train_batches'] = [TensorKey(tensor_name,'GLOBAL',0,('model',)) for tensor_name in tensor_names]

        #Validation may be performed on local or aggregated (global) model, so there is an extra lookup dimension for kwargs
        self.required_tensorkeys_for_function['validate'] = {}
        #TODO This is not stateless. The optimizer will not be
        self.required_tensorkeys_for_function['validate']['local_model=True'] = \
                [TensorKey(tensor_name,'LOCAL',0,('trained',)) for tensor_name in tensor_names]
        self.required_tensorkeys_for_function['validate']['local_model=False'] = \
                [TensorKey(tensor_name,'GLOBAL',0,('model',)) for tensor_name in tensor_names] 




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

