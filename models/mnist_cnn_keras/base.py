"""Base classes for developing a Federated Learning model.

You may copy this file as the starting point of your own model.
"""
import logging
import numpy as np
import tensorflow as tf
import tqdm

import tensorflow.keras as keras
from tensorflow.keras import backend as K
from .export_init_weights import export_weights

class FLModel(object):
    """An abstract class used to represent a model training procedure."""
    def __init__(self, *args, **argv):
        super(FLModel, self).__init__()
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


class FLKerasModel(FLModel):
    """A class used to represent a Keras model training procedure."""
    def __init__(self, data, *args, **argv):
        super(FLKerasModel, self).__init__(*args, **argv)
        self.model = keras.Model()
        self.data = data
        NUM_PARALLEL_EXEC_UNITS = 1
        config = tf.ConfigProto(intra_op_parallelism_threads=NUM_PARALLEL_EXEC_UNITS, inter_op_parallelism_threads=1, allow_soft_placement=True, device_count = {'CPU': NUM_PARALLEL_EXEC_UNITS })
        self.sess = tf.Session(config=config)
        K.set_session(self.sess)

    def get_data(self):
        return self.data

    def set_data(self, data):
        if data.get_feature_shape() != self.data.get_feature_shape():
            raise ValueError('Data feature shape is not compatible with model.')
        self.data = data

    def train_epoch(self):
        self.is_initial = False
        history = self.model.fit(self.data.X_train, self.data.y_train,
          batch_size=self.data.batch_size,
          epochs=1,
          verbose=0,)
        # As we alwasy train one epoch, we only need the first element in the list.
        ret_dict = {name:values[0] for name, values in history.history.items()}
        return ret_dict['loss']

    def validate(self):
        vals = self.model.evaluate(self.data.X_val, self.data.y_val, verbose=0)
        metrics_names = self.model.metrics_names
        ret_dict = dict(zip(metrics_names, vals))
        return ret_dict['acc']

    def get_training_data_size(self):
        return self.data.get_training_data_size()

    def get_validation_data_size(self):
        return self.data.get_validation_data_size()

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

        if with_opt_vars is True:
            opt_weights = self._get_weights_dict(self.model.optimizer)
            model_weights.update(opt_weights)
            if len(opt_weights) == 0:
                self.logger.debug("WARNING: We didn't find variables for the optimizer.")
        return model_weights

    def set_tensor_dict(self, tensor_dict, with_opt_vars):
        self.is_initial = False

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

    
    def export_init_weights(self, model_name, version, fpath):
        export_weights(model_name=model_name, 
                       version=version, 
                       tensor_dict=self.get_tensor_dict(with_opt_vars=False), 
                       fpath=fpath)
