# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""MLCubeTaskRunner module."""

import numpy as np
import tensorflow as tf
import subprocess
import json

from openfl.utilities import split_tensor_dict_for_holdouts
from openfl.utilities import TensorKey
from .runner import TaskRunner
from .runner_keras import KerasTaskRunner
from .runner_pt import PyTorchTaskRunner


class MLCubeTaskRunner(TaskRunner):
    """A wrapper for MLCube Tasks."""

    def __init__(self, mlcube_runner_type='docker', model_type='pytorch', **kwargs):
        """Initialize.

        Args:
            TBD
        """
        super().__init__(**kwargs)
        if model_type == 'pytorch':
            impl = PyTorchTaskRunner
        elif model_type == 'tensorflow':
            impl = KerasTaskRunner

        # Need to call dummy train task to load initial model
        self.dummy_train()
        self.load_native('model')

        self.optimizer = self.model.optimizer
        self.runner = impl(**kwargs)
        self.runner.model = self.model
        self.runner.optimizer = self.optimizer
        self.required_tensorkeys_for_function = {}
        self.tensor_dict_split_fn_kwargs = \
            self.runner.tensor_dict_split_fn_kwargs
        self.initialize_tensorkeys_for_functions()

    def dummy_train(self):
        """
        Perform dummy training MLCube task which serialize the model to disk
        """
        dummy_train_proc = subprocess.run("mlcube_docker","run","--mlcube=.","--platform=platforms/docker.yaml","--task=run/dummy_train.yaml")

    def train(self, col_name, round_num, input_tensor_dict, epochs, **kwargs):
        """Perform training for a specified number of epochs."""
        if 'metrics' not in kwargs:
            raise KeyError('metrics must be included in kwargs')
        param_metrics = kwargs['metrics']

        self.rebuild_model(round_num, input_tensor_dict)

        # 1. Save model in native format
        self.save_native('model')

        # 2. Call MLCube train task
        train_proc = subprocess.run("mlcube_docker","run","--mlcube=.","--platform=platforms/docker.yaml","--task=run/train.yaml")

        # 3. Load model from native format
        self.load_native('model')
        # 4. Load any metrics
        metrics = self.load_metrics('metrics')
        # 5. Convert to tensorkeys


        # output metric tensors (scalar)
        origin = col_name
        tags = ('trained',)
        output_metric_dict = {
            TensorKey(
                metric_name, origin, round_num, True, ('metric',)
            ): np.array(
                    metrics[metric_name]
                ) for metric_name in metrics}

        # output model tensors (Doesn't include TensorKey)
        output_model_dict = self.get_tensor_dict(with_opt_vars=True)
        global_model_dict, local_model_dict = split_tensor_dict_for_holdouts(
            self.logger, output_model_dict,
            **self.tensor_dict_split_fn_kwargs
        )

        # create global tensorkeys
        global_tensorkey_model_dict = {
            TensorKey(
                tensor_name, origin, round_num, False, tags
            ): nparray for tensor_name, nparray in global_model_dict.items()
        }
        # create tensorkeys that should stay local
        local_tensorkey_model_dict = {
            TensorKey(
                tensor_name, origin, round_num, False, tags
            ): nparray for tensor_name, nparray in local_model_dict.items()
        }
        # the train/validate aggregated function of the next round will look
        # for the updated model parameters.
        # this ensures they will be resolved locally
        next_local_tensorkey_model_dict = {
            TensorKey(
                tensor_name, origin, round_num + 1, False, ('model',)
            ): nparray for tensor_name, nparray in local_model_dict.items()
        }

        global_tensor_dict = {
            **output_metric_dict,
            **global_tensorkey_model_dict
        }
        local_tensor_dict = {
            **local_tensorkey_model_dict,
            **next_local_tensorkey_model_dict
        }

        # update the required tensors if they need to be pulled from the
        # aggregator
        # TODO this logic can break if different collaborators have different
        #  roles between rounds.
        # for example, if a collaborator only performs validation in the first
        # round but training in the second, it has no way of knowing the
        # optimizer state tensor names to request from the aggregator
        # because these are only created after training occurs.
        # A work around could involve doing a single epoch of training
        # on random data to get the optimizer names, and then throwing away
        # the model.
        if self.opt_treatment == 'CONTINUE_GLOBAL':
            self.initialize_tensorkeys_for_functions(with_opt_vars=True)

        return global_tensor_dict, local_tensor_dict

    def validate(self, col_name, round_num, input_tensor_dict, **kwargs):
        """
        Run the trained model on validation data; report results.

        Parameters
        ----------
        input_tensor_dict : either the last aggregated or locally trained model

        Returns
        -------
        output_tensor_dict : {TensorKey: nparray} (these correspond to acc,
         precision, f1_score, etc.)
        """
        self.rebuild_model(round_num, input_tensor_dict, validation=True)
        param_metrics = kwargs['metrics']

        # 2. Call MLCube validate task
        train_proc = subprocess.run("mlcube_docker","run","--mlcube=.","--platform=platforms/docker.yaml","--task=run/evaluate.yaml")

        # 3. Load any metrics
        metrics = self.load_metrics('metrics')

        # 4. Convert to tensorkeys
    
        origin = col_name
        suffix = 'validate'
        if kwargs['apply'] == 'local':
            suffix += '_local'
        else:
            suffix += '_agg'
        tags = ('metric', suffix)
        output_tensor_dict = {
            TensorKey(
                metric_name, origin, round_num, True, tags
            ): np.array(metrics[metric_name])
            for metric_name in metrics
        }

        return output_tensor_dict, {}

    def initialize_tensorkeys_for_functions(self, with_opt_vars=False):
        """
        Set the required tensors for all publicly accessible methods that could \
            be called as part of a task.

        By default, this is just all of the layers and optimizer of the model.
         Custom tensors should be added to this function

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Call dummy train function
        self.runner.initialize_tensorkeys_for_functions(with_opt_vars)

    def build_model(self):
        """Abstract method."""
        raise NotImplementedError

    def get_required_tensorkeys_for_function(self, func_name, **kwargs):
        """
        When running a task, a map of named tensorkeys must be provided to the \
            function as dependencies.

        Returns:
            list: (TensorKey(tensor_name, origin, round_number))
        """
        return self.runner.get_required_tensorkeys_for_function(
            func_name, **kwargs)

    def get_tensor_dict(self, with_opt_vars):
        """
        Get the weights.

        Args:
            with_opt_vars (bool): Specify if we also want to get the variables
             of the optimizer.

        Returns:
            dict: The weight dictionary {<tensor_name>: <value>}
        """
        return self.runner.get_tensor_dict(with_opt_vars)

    def set_tensor_dict(self, tensor_dict, with_opt_vars):
        """
        Set the model weights with a tensor dictionary: {<tensor_name>: <value>}.

        Args:
            tensor_dict (dict): The model weights dictionary.
            with_opt_vars (bool): Specify if we also want to set the variables
            of the optimizer.

        Returns:
            None
        """
        return self.runner.set_tensor_dict(tensor_dict, with_opt_vars)

    def reset_opt_vars(self):
        """Reinitialize the optimizer variables."""
        return self.runner.reset_opt_vars()

    def initialize_globals(self):
        """
        Initialize all global variables.

        Returns:
            None
        """
        return self.runner.initialize_globals()

    def load_metrics(self, filepath):
        """
        Load metrics from JSON file
        """
        ### 
        with open(filepath) as json_file:
                metrics = json.load(json_file)
        return metrics

    def load_native(self, filepath, **kwargs):
        """
        Load model state from a filepath in ML-framework "native" format, \
            e.g. PyTorch pickled models.

        May load from multiple files. Other filepaths may be derived from the
        passed filepath, or they may be in the kwargs.

        Args:
            filepath (string): Path to frame-work specific file to load. For
                               frameworks that use multiple files, this string
                               must be used to derive the other filepaths.
            kwargs           : For future-proofing

        Returns:
            None
        """
        return self.runner.load_native(filepath, **kwargs)

    def save_native(self, filepath, **kwargs):
        """
        Save model state in ML-framework "native" format, \
            e.g. PyTorch pickled models.

        May save one file or multiple files, depending on the framework.

        Args:
            filepath (string): If framework stores a single file, this should
                               be a single file path.
            Frameworks that store multiple files may need to derive the other
            paths from this path.
            kwargs           : For future-proofing

        Returns:
            None
        """
        return self.runner.save_native(filepath, **kwargs)

    def rebuild_model(self, round_num, input_tensor_dict, validation=False):
        """
        Parse tensor names and update weights of model. Handles the optimizer treatment.

        Returns:
            None
        """
        return self.runner.rebuild_model(
            round_num, input_tensor_dict, validation)

    def set_optimizer_treatment(self, opt_treatment):
        """Change treatment of current instance optimizer."""
        super().set_optimizer_treatment(opt_treatment)
        self.runner.opt_treatment = opt_treatment
