# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""You may copy this file as the starting point of your own model."""

import tensorflow.keras as keras
import numpy as np

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from openfl.federated.data import FederatedDataLoader


class KerasCNN:
    """A basic convolutional neural network model."""

    def __init__(self, **kwargs):
        """
        Initialize.

        Args:
            **kwargs: Additional parameters to pass to the function
        """

        x_train, y_train, x_test, y_test = self.setup_data()

        self.x_train = FederatedDataLoader(x_train)
        self.y_train = FederatedDataLoader(y_train)
        self.x_test = FederatedDataLoader(x_test)
        self.y_test = FederatedDataLoader(y_test)

        self.model = self.build_model()

        self.model.summary()

    def setup_data(self):
        # Define data loader
        num_classes = 10
        input_shape = (28, 28, 1)

        # the data, split between train and test sets
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

        # Scale images to the [0, 1] range
        x_train = x_train.astype("float32") / 255
        x_test = x_test.astype("float32") / 255
        # Make sure images have shape (28, 28, 1)
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)
        print("x_train shape:", x_train.shape)
        print(x_train.shape[0], "train samples")
        print(x_test.shape[0], "test samples")

        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)

        return x_train,y_train,x_test,y_test



    def build_model(self,
                    input_shape=(28,28,1),
                    num_classes=10,
                    conv_kernel_size=(4, 4),
                    conv_strides=(2, 2),
                    conv1_channels_out=16,
                    conv2_channels_out=32,
                    final_dense_inputsize=100,
                    **kwargs):
        """
        Define the model architecture.

        Args:
            input_shape (numpy.ndarray): The shape of the data
            num_classes (int): The number of classes of the dataset

        Returns:
            tensorflow.python.keras.engine.sequential.Sequential: The model defined in Keras

        """
        model = Sequential()

        model.add(Conv2D(conv1_channels_out,
                         kernel_size=conv_kernel_size,
                         strides=conv_strides,
                         activation='relu',
                         input_shape=input_shape))

        model.add(Conv2D(conv2_channels_out,
                         kernel_size=conv_kernel_size,
                         strides=conv_strides,
                         activation='relu'))

        model.add(Flatten())

        model.add(Dense(final_dense_inputsize, activation='relu'))

        model.add(Dense(num_classes, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])

        # initialize the optimizer variables
        opt_vars = model.optimizer.variables()

        for v in opt_vars:
            v.initializer.run(session=self.sess)

        return model

    def train(self, epochs=1, batch_size=32, metric='loss'):
        """
        Perform the training for a specified number of batches.

        Is expected to perform draws randomly, without replacement until data is exausted.
        Then data is replaced and shuffled and draws continue.

        Returns
        -------
            
        """

        history = self.model.fit(self.x_train,
                                 self.y_train,
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 verbose=0, )


        loss = np.mean([history.history[metric]])
        return loss 

    def validate(self,batch_size=32):
        """
        Run the trained model on validation data; report results.

        Parameters
        ----------
        input_tensor_dict : either the last aggregated or locally trained model

        Returns
        -------
        metric : float (accuracy, f1_score, etc.)
        """

        vals = self.model.evaluate(
            self.x_test,
            self.y_test,
            batch_size=batch_size,
            verbose=0
        )

        model_metrics_names = self.model.metrics_names
        if type(vals) is not list:
            vals = [vals]
        ret_dict = dict(zip(model_metrics_names, vals))
        accuracy = ret_dict['accuracy']

        return accuracy


