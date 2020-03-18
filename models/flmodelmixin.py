"""Mixin class for FL models. No default implementation.

Each framework will likely have its own baseclass implementation (e.g. TensorflowFLModelBase) that uses this mixin.

You may copy use this file or the appropriate framework-specific base-class to port your own models.
"""
import logging


class FLModelMixin(object):
    def set_logger(self):
        self.logger = logging.getLogger(__name__)

    def get_data(self):
        """
        Get the data object.
        Serves up batches and provides infor regarding data.

        Returns
        -------
        data object
        """
        # TODO: establish class inheritance for data object for guidance.
        raise NotImplementedError

    def set_data(self, data):
        """
        Set data object.

        Returns
        -------
        None
        """
        raise NotImplementedError

    def train_epoch(self):
        """
        Train one epoch.

        Returns
        -------
        dict
            {<metric>: <value>}
        """
        raise NotImplementedError

    def get_training_datasize(self):
        """
        Get the number of training examples.
        It will be used for weighted averaging in aggregation.

        Returns
        -------
        int
            The number of training examples.
        """
        raise NotImplementedError

    def validate(self):
        """
        Run validation.

        Returns
        -------
        dict
            {<metric>: <value>}
        """
        raise NotImplementedError

    def get_validation_data_size(self):
        """
        Get the number of examples.
        It will be used for weighted averaging in aggregation.

        Returns
        -------
        int
            The number of validation examples.
        """
        raise NotImplementedError
        
    def initialize_globals(self):
        """
        Initialize all global variables
        ----------

        Returns
        -------
        None
        """
        raise NotImplementedError

    def get_tensor_dict(self, with_opt_vars):
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
        raise NotImplementedError

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
        raise NotImplementedError

    def reset_opt_vars(self):
        """Reinitialize the optimizer variables."""
        raise NotImplementedError
