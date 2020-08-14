# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

"""Base classes for developing a keras.Model() Federated Learning model.

You may copy this file as the starting point of your own keras model.
"""
import logging
import numpy as np
import tqdm
import tensorflow as tf

from models import FLModel
from tfedlrn import TensorKey,split_tensor_dict_for_holdouts

import tensorflow.keras as keras
from tensorflow.keras import backend as K

class KerasFLModel(FLModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.model = keras.Model()

        self.model_tensor_names = []

        #This is a map of all of the required tensors for each of the public functions in KerasFLModel
        self.required_tensorkeys_for_function = {}

        NUM_PARALLEL_EXEC_UNITS = 1
        config = tf.ConfigProto(intra_op_parallelism_threads=NUM_PARALLEL_EXEC_UNITS, 
                                inter_op_parallelism_threads=1, 
                                allow_soft_placement=True, 
                                device_count = {'CPU': NUM_PARALLEL_EXEC_UNITS })
        config.gpu_options.allow_growth=True
        
        self.sess = tf.Session(config=config)
        K.set_session(self.sess)

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


    def train(self, col_name, round_num, input_tensor_dict, epochs, **kwargs):
        """
        Perform the training for a specified number of batches. Is expected to perform draws randomly, without 
        replacement until data is exausted. Then data is replaced and shuffled and draws continue.

        Returns
        -------
        dict
            'TensorKey: nparray'
        """
        if 'metrics' not in kwargs:
            raise KeyError('metrics must be included in kwargs')
        if 'batch_size' in kwargs:
            batch_size = kwargs['batch_size']
        else:
            batch_size = self.data.batch_size

        #rebuild model with updated weights
        self.rebuild_model(round_num, input_tensor_dict)

        history = self.model.fit(self.data.X_train, 
                                 self.data.y_train,
                                 batch_size=self.data.batch_size,
                                 epochs=epochs,
                                 verbose=0,)

        #TODO Currently assuming that all metrics are defined at initialization (build_model). If metrics are added (i.e.
        # not a subset of what was originally defined) then the model must be recompiled. 
        model_metrics_names = self.model.metrics_names
        param_metrics = kwargs['metrics']
        #TODO if there are new metrics in the flplan that were not included in the originally compiled model, that behavior
        #is not currently handled. 
        for param in param_metrics:
            if param not in model_metrics_names:
                error = 'KerasFLModel does not support specifying new metrics. Param_metrics = {}, model_metrics_names = {}'.format(param_metrics,model_metrics_names)
                raise ValueError(error)

        #Output metric tensors (scalar)
        origin = col_name
        tags = ('trained',)
        output_metric_dict = {TensorKey(metric,origin,round_num,True,('metric',)): np.array(np.mean([history.history[metric]])) for metric in param_metrics}

        #output model tensors (Doesn't include TensorKey)
        output_model_dict = self.get_tensor_dict(with_opt_vars=True)
        global_model_dict,local_model_dict = split_tensor_dict_for_holdouts(self.logger, output_model_dict, **self.tensor_dict_split_fn_kwargs)

        #Create global tensorkeys
        global_tensorkey_model_dict = {TensorKey(tensor_name,origin,round_num,False,tags): nparray for tensor_name,nparray in global_model_dict.items()}
        #Create tensorkeys that should stay local
        local_tensorkey_model_dict = {TensorKey(tensor_name,origin,round_num,False,tags): nparray for tensor_name,nparray in local_model_dict.items()}
        #The train/validate aggregated function of the next round will look for the updated model parameters. 
        #This ensures they will be resolved locally
        next_local_tensorkey_model_dict = {TensorKey(tensor_name,origin,round_num+1,False,('model',)): nparray for tensor_name,nparray in local_model_dict.items()}


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




    def validate(self, col_name, round_num, input_tensor_dict,**kwargs):
        """
        Run the trained model on validation data; report results
        Parameters
        ----------
        input_tensor_dict : either the last aggregated or locally trained model

        Returns
        -------
        output_tensor_dict : {TensorKey: nparray} (these correspond to acc, precision, f1_score, etc.)
        """

        batch_size = 1
        if 'batch_size' in kwargs:
            batch_size = kwargs['batch_size']
        self.rebuild_model(round_num, input_tensor_dict)
        param_metrics = kwargs['metrics']

        vals = self.model.evaluate(self.data.X_val, self.data.y_val,batch_size=batch_size, verbose=0)
        model_metrics_names = self.model.metrics_names
        ret_dict = dict(zip(model_metrics_names, vals))

        #TODO if there are new metrics in the flplan that were not included in the originally compiled model, that behavior
        #is not currently handled. 
        for param in param_metrics:
            if param not in model_metrics_names:
                error = 'KerasFLModel does not support specifying new metrics. Param_metrics = {}, model_metrics_names = {}'.format(param_metrics,model_metrics_names)
                raise ValueError(error)
        
        origin = col_name 
        suffix = 'validate'
        if kwargs['apply'] == 'local':
            suffix += '_local'
        else:
            suffix += '_agg'
        tags = ('metric',suffix)
        output_tensor_dict = {TensorKey(metric,origin,round_num,True,tags): np.array(ret_dict[metric]) for metric in param_metrics}

        return output_tensor_dict,{}

    @staticmethod
    def _get_weights_names(obj):
        """
        Get the list of weight names.
        Parameters
        ----------
        obj : Model or Optimizer
            The target object that we want to get the weights.

        Returns
        -------
        dict
            The weight name list
        """

        weight_names = [weight.name for weight in obj.weights]
        return weight_names


    @staticmethod
    def _get_weights_dict(obj,suffix=''):
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
            weights_dict[name+suffix] = value
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

    def get_tensor_dict(self, with_opt_vars, suffix=''):
        """
        Get the model weights as a tensor dictionary.

        Parameters
        ----------
        with_opt_vars : bool
            If we should include the optimizer's status.
        suffix : string 
            Universally 

        Returns
        -------
        dict
            The tensor dictionary.
        """
        model_weights = self._get_weights_dict(self.model,suffix)

        if with_opt_vars:
            opt_weights = self._get_weights_dict(self.model.optimizer,suffix)

            model_weights.update(opt_weights)
            if len(opt_weights) == 0:
                self.logger.debug("WARNING: We didn't find variables for the optimizer.")
        return model_weights

    def set_tensor_dict(self, tensor_dict, with_opt_vars):
        
        if with_opt_vars is False:
            #self._set_weights_dict(self.model, tensor_dict)
            #It is possible to pass in opt variables from the input tensor dict
            #This will make sure that the correct layers are updated
            model_weight_names = [weight.name for weight in self.model.weights]
            model_weights_dict = {name: tensor_dict[name] for name in model_weight_names}
            self._set_weights_dict(self.model, model_weights_dict)
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

    def set_required_tensorkeys_for_function(self, func_name, tensor_key, **kwargs):
        """
        Set the required tensors for specified function that could be called as part of a task.
        By default, this is just all of the layers and optimizer of the model. Custom tensors should be added to this function

        Parameters
        ----------
        func_name: string
        tensor_key: TensorKey (namedtuple)
        **kwargs: Any function arguments {}

        Returns
        -------
        None
        """

        #TODO there should be a way to programmatically iterate through all of the methods in the class and declare the tensors.
        #For now this is done manually

        if func_name == 'validate':
            #Should produce 'apply=global' or 'apply=local'
            local_model = 'apply' + kwargs['apply']
            self.required_tensorkeys_for_function[func_name][local_model].append(tensor_key)
        else:
            self.required_tensorkeys_for_function[func_name].append(tensor_key)

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
            local_model = 'apply=' + str(kwargs['apply'])
            return self.required_tensorkeys_for_function[func_name][local_model]
        else:
            return self.required_tensorkeys_for_function[func_name]


    def update_tensorkeys_for_functions(self):
        """
        Update the required tensors for all publicly accessible methods that could be called as part of a task.
        By default, this is just all of the layers and optimizer of the model. Custom tensors should be added to this function

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        
        #TODO complete this function. It is only needed for opt_treatment, and making the model stateless

        #Minimal required tensors for train function 
        model_layer_names = self._get_weights_names(self.model)
        opt_names = self._get_weights_names(self.model.optimizer)
        tensor_names = model_layer_names + opt_names
        self.logger.debug('Updating model tensor names: {}'.format(tensor_names))
        self.required_tensorkeys_for_function['train'] = [TensorKey(tensor_name,'GLOBAL',0,('model',)) for tensor_name in tensor_names]

        #Validation may be performed on local or aggregated (global) model, so there is an extra lookup dimension for kwargs
        self.required_tensorkeys_for_function['validate'] = {}
        self.required_tensorkeys_for_function['validate']['local_model=True'] = \
                [TensorKey(tensor_name,'LOCAL',0,('trained',)) for tensor_name in tensor_names]
        self.required_tensorkeys_for_function['validate']['local_model=False'] = \
                [TensorKey(tensor_name,'GLOBAL',0,('model',)) for tensor_name in tensor_names]


    def initialize_tensorkeys_for_functions(self):
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

        #TODO there should be a way to programmatically iterate through all of the methods in the class and declare the tensors.
        #For now this is done manually

        #Minimal required tensors for train function 
        tensor_names = self._get_weights_names(self.model) + self._get_weights_names(self.model.optimizer)
        self.logger.debug('Initial model tensor names: {}'.format(tensor_names))
        self.required_tensorkeys_for_function['train'] = [TensorKey(tensor_name,'GLOBAL',0,False,('model',)) for tensor_name in tensor_names]

        #Validation may be performed on local or aggregated (global) model, so there is an extra lookup dimension for kwargs
        self.required_tensorkeys_for_function['validate'] = {}
        #TODO This is not stateless. The optimizer will not be 
        self.required_tensorkeys_for_function['validate']['apply=local'] = \
                [TensorKey(tensor_name,'LOCAL',0,False,('trained',)) for tensor_name in tensor_names]
        self.required_tensorkeys_for_function['validate']['apply=global'] = \
                [TensorKey(tensor_name,'GLOBAL',0,False,('model',)) for tensor_name in tensor_names]

