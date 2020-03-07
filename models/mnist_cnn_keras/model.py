import logging
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from .base import FLKerasModel

class ConvModel(FLKerasModel):
    """A convolutional neural network model for MNIST.

    """
    def __init__(self, data, **kwargs):
        super(ConvModel, self).__init__(data=data)
        self.logger = logging.getLogger(__name__)
        self.model = self.build_model(data.get_feature_shape(), data.num_classes)
        print(self.model.summary())
        if self.data.y_train is not None and self.data.y_val is not None:
            print("Training set size: %d; Validation set size: %d" % (len(self.data.y_train), len(self.data.y_val)))

        self.is_initial = True
        self.initial_opt_weights = self._get_weights_dict(self.model.optimizer)


    @staticmethod
    def build_model(input_shape, num_classes):
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


    
