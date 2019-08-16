import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from ..base import FLKerasModel

from tensorflow.keras.datasets import mnist


class ConvModel(FLKerasModel):
    """A convolutional neural network model for MNIST."""
    def __init__(self, *args, **argv):
        super(ConvModel, self).__init__(*args, **argv)
        self.batch_size = 32
        input_shape, num_classes, self.x_train, self.y_train, self.x_test, self.y_test = self.load_dataset()
        self.model = self.build_model(input_shape, num_classes)
        print(self.model.summary())

        self.is_initial = True

        self.initial_opt_weights = self._get_weights_dict(self.model.optimizer)
    
    @staticmethod
    def load_dataset():
        """
        Load the MNIST dataset.

        Returns
        -------
        list
            The input shape.
        int
            The number of classes.
        numpy.ndarray
            The training data.
        numpy.ndarray
            The training labels.
        numpy.ndarray
            The validation data.
        numpy.ndarray
            The validation labels.
        """
        img_rows, img_cols = 28, 28
        num_classes = 10
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
            input_shape = (1, img_rows, img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
            input_shape = (img_rows, img_cols, 1)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        print('x_train shape:', x_train.shape)
        print('y_train shape:', y_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')

        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)

        return input_shape, num_classes, x_train, y_train, x_test, y_test


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
