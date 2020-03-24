# Copyright (C) 2020 Intel Corporation
# Licensed subject to Collaboration Agreement dated February 28th, 2020 between Intel Corporation and Trustees of the University of Pennsylvania.

import logging
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from models.tensorflow import KerasFLModel

class KerasCNN(KerasFLModel):
    """A basic convolutional neural network model.

    """
    def __init__(self, data, **kwargs):
        super().__init__(data)
        
        self.model = self.build_model(self.feature_shape, data.num_classes, **kwargs)
        
        print(self.model.summary())
        if self.data is not None:
            print("Training set size: %d; Validation set size: %d" % (self.get_training_data_size, self.get_validation_data_size))

    @staticmethod
    def build_model(input_shape, num_classes, **kwargs):
        """
        Define the model architecture.
        Parameters
        ----------
        input_shape : numpy.ndarray
            The shape of the data.
        num_classes : int
            The number of classes of the dataset.

        Returns
        -------
        tensorflow.python.keras.engine.sequential.Sequential
            The model defined in Keras.

        """
        model = Sequential()
        model.add(Conv2D(16,
                        kernel_size=(4, 4),
                        strides=(2,2),
                        activation='relu',
                        input_shape=input_shape))
        model.add(Conv2D(32,
                        kernel_size=(4, 4),
                        strides=(2,2),
                        activation='relu'))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                        optimizer=keras.optimizers.Adam(),
                        metrics=['accuracy'])
        return model


    
