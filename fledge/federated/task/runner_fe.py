# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

import numpy         as np
import torch         as pt
import torch.nn      as nn
import fastestimator as fe

from copy import deepcopy

from fledge.utilities import TensorKey, split_tensor_dict_for_holdouts

from .runner_keras import KerasTaskRunner
from .runner_pt    import PyTorchTaskRunner

class FastEstimatorTaskRunner(KerasTaskRunner,PyTorchTaskRunner):
    """FastEstimator class for Federated Learning
    """

    def __init__(self, **kwargs):
        """Initializer

        Args:
            **kwargs: Additional parameters to pass to the functions
        """

        #super().__init__(**kwargs)

        #Keep reference to Pytorch, Keras TaskRunners to initialize when model is actually known
        self.ktr = KerasTaskRunner
        self.pttr = PyTorchTaskRunner
        self.kwargs = kwargs
        
        self.estimator = None

    def set_runner_type(self,model):
        """
        Inherit methods from Pytorch or Keras depending on model type
        """
        kwargs = self.kwargs
        if 'keras' in str(type(model)):
            #The keras task runner reinitializes the model. Save a reference to the compiled model and overwrite self.model
            model = self.model
            self.ktr.__init__(self,**kwargs)
            self.model = model
        if isinstance(model,nn.Module):
            self.pttr.__init__(self,**kwargs)
        

    def train(self, col_name, round_num, input_tensor_dict,epochs, **kwargs):
        """
        Perform training for a specified number of epochs
        """

        if 'metrics' not in kwargs:
            raise KeyError('metrics must be included in kwargs')
        param_metrics = kwargs['metrics']

        self.rebuild_model(round_num, input_tensor_dict)

        #Estimators need to be given an experiment name to produce an output summary
        summary = self.estimator.fit("experiment")

        #Define what the ouptut is to encapsulate in tensorkeys and return
        # output metric tensors (scalar)
        origin = col_name
        tags   = ('trained',)
        output_metric_dict = {TensorKey(metric,origin,round_num,True,('metric',)): \
                np.array(list(summary.history['train'][metric].values())[-1]) for metric in param_metrics}

      # output model tensors (Doesn't include TensorKey)
        output_model_dict = self.get_tensor_dict(with_opt_vars=True)
        global_model_dict,local_model_dict = split_tensor_dict_for_holdouts(self.logger, output_model_dict, **self.tensor_dict_split_fn_kwargs)

      # create global tensorkeys
        global_tensorkey_model_dict = {TensorKey(tensor_name, origin, round_num, False, tags): nparray for tensor_name,nparray in global_model_dict.items()}
      # create tensorkeys that should stay local
        local_tensorkey_model_dict  = {TensorKey(tensor_name, origin, round_num, False, tags): nparray for tensor_name,nparray in local_model_dict.items()}
      # the train/validate aggregated function of the next round will look for the updated model parameters. 
      # this ensures they will be resolved locally
        next_local_tensorkey_model_dict = {TensorKey(tensor_name, origin,round_num + 1, False,('model',)): nparray for tensor_name,nparray in local_model_dict.items()}


        global_tensor_dict = {**output_metric_dict, **global_tensorkey_model_dict}
        local_tensor_dict  = {**local_tensorkey_model_dict, **next_local_tensorkey_model_dict}

        #update the required tensors if they need to be pulled from the aggregator
        #TODO this logic can break if different collaborators have different roles between rounds.
        #for example, if a collaborator only performs validation in the first round but training
        #in the second, it has no way of knowing the optimizer state tensor names to request from the aggregator
        #because these are only created after training occurs. A work around could involve doing a single epoch of training
        #on random data to get the optimizer names, and then throwing away the model.
        if  self.opt_treatment == 'CONTINUE_GLOBAL':
            self.initialize_tensorkeys_for_functions(with_opt_vars = True)

      # return global_tensor_dict, local_tensor_dict
        return global_tensor_dict,local_tensor_dict

    def validate(self, col_name, round_num, input_tensor_dict, **kwargs):
        """
        Run the trained model on validation data; report results
        Parameters
        ----------
        input_tensor_dict : either the last aggregated or locally trained model

        Returns
        -------
        output_tensor_dict : {TensorKey: nparray} (these correspond to acc, precision, f1_score, etc.)
        """

        self.rebuild_model(round_num, input_tensor_dict,validation = True)
        param_metrics = kwargs['metrics']

        results = self.estimator.test('experiment')
        ret_dict = {metric: list(results.history['test'][metric].values())[-1] for metric in param_metrics}

        origin = col_name 
        suffix = 'validate'
        if  kwargs['apply'] == 'local':
            suffix += '_local'
        else:
            suffix += '_agg'
        tags = ('metric',suffix)
        output_tensor_dict = {TensorKey(metric,origin,round_num,True,tags): np.array(ret_dict[metric]) for metric in param_metrics}

        return output_tensor_dict,{}


    def initialize_tensorkeys_for_functions(self, with_opt_vars = False):
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

        output_model_dict = self.get_tensor_dict(with_opt_vars=with_opt_vars)
        global_model_dict,local_model_dict = split_tensor_dict_for_holdouts(self.logger, output_model_dict, **self.tensor_dict_split_fn_kwargs)
        if not with_opt_vars:
            validation_global_model_dict = global_model_dict
            validation_local_model_dict = local_model_dict
        else:
            output_model_dict = self.get_tensor_dict(with_opt_vars=False)
            validation_global_model_dict, validation_local_model_dict = split_tensor_dict_for_holdouts(self.logger, output_model_dict, **self.tensor_dict_split_fn_kwargs)


        self.required_tensorkeys_for_function['train'] = [TensorKey(tensor_name,'GLOBAL',0,False,('model',)) for tensor_name in global_model_dict]
        self.required_tensorkeys_for_function['train'] += [TensorKey(tensor_name,'LOCAL',0,False,('model',)) for tensor_name in local_model_dict]


        #Validation may be performed on local or aggregated (global) model, so there is an extra lookup dimension for kwargs
        self.required_tensorkeys_for_function['validate'] = {}
        #TODO This is not stateless. The optimizer will not be 
        self.required_tensorkeys_for_function['validate']['apply=local'] = \
                [TensorKey(tensor_name,'LOCAL',0,False,('trained',)) for tensor_name in {**validation_global_model_dict,**validation_local_model_dict}]
        self.required_tensorkeys_for_function['validate']['apply=global'] = \
                [TensorKey(tensor_name,'GLOBAL',0,False,('model',)) for tensor_name in validation_global_model_dict]
        self.required_tensorkeys_for_function['validate']['apply=global'] += \
                [TensorKey(tensor_name,'LOCAL',0,False,('model',)) for tensor_name in validation_local_model_dict]

