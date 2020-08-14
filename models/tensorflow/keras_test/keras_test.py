# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

import logging
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from models.tensorflow import KerasFLModel
from tfedlrn import TensorKey

class KerasTest(KerasFLModel):
    """An empty model that has custom functions.

    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.model = self.build_model()

        self.initialize_tensorkeys_for_functions()
        self.initialize_tensorkeys_for_custom_functions()
        self.internal_count = np.random.randint(10)

        self.set_logger()

    def build_model(self):
        """
        Empty model architecture.
        """
        model = Sequential()
        return model


    def dummy_function(self, col_name, round_num, input_tensor_dict,metric_name_prefix='dummy',**kwargs):
        """This function is purely an example of the types of things you can do with the low level API. In this case, it just increments
           its internal count and reports it to the aggregator"""
        #Do nothing with the input_tensor_dict
        output_tensor_dict = {TensorKey('{}:internal count'.format(metric_name_prefix),col_name,round_num,True,('metric',)):np.array(self.internal_count)}
        self.internal_count += 1
        return output_tensor_dict,{}


    def initialize_tensorkeys_for_custom_functions(self): 
        """
        For testing purposes, dummy function doesn't have any dependencies
        
        Returns
        -------
        []
        """
        self.required_tensorkeys_for_function['dummy_function'] = []

