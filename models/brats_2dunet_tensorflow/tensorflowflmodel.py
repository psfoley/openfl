import os
from math import ceil

import numpy as np
import tensorflow as tf
import abc

from tfedlrn.collaborator.flmodel import FLModel

class TensorFlowFLModel(FLModel):
    """WIP code. Goal is to simplify porting a model to this framework.
    Currently, this creates a placeholder and assign op for every variable, which grows the graph considerably.
    Also, the abstraction for the tf.session isn't ideal yet."""

    def __init__(self):
        self.assign_ops = None
        self.placeholders = None

    @abc.abstractmethod
    def get_session(self):
        pass

    @abc.abstractmethod
    def get_vars(self):
        pass

    def get_tensor_dict(self):
        """FIXME: how to protect against redundant graphs?"""        
        return {v.name: self.get_session().run(v) for v in self.get_vars()}

    def set_tensors_from_dict(self, tensor_dict):
        """FIXME: how to protect against redundant graphs?"""
        if self.placeholders is None:
            self.placeholders = {v.name: tf.placeholder(v.dtype, shape=v.shape) for v in self.get_vars()}
        if self.assign_ops is None:
            self.assign_ops = {v.name: tf.assign(v, self.placeholders[v.name]) for v in self.get_vars()}

        session = self.get_session()
        for k, v in tensor_dict:
            session.run(self.assign_ops[k], feed_dict={self.placeholders[k]:v})
