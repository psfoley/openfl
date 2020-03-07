import numpy as np
from math import ceil

import tensorflow.keras as keras
from tensorflow.keras import backend as K


def load_dataset(raw_path):
        """
        Load the MNIST dataset.

        Params
        ------
        raw_path: str
            The path to the raw npz file.

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
        def _load_raw_dataset(path):
            with np.load(path) as f:
                x_train, y_train = f['x_train'], f['y_train']
                x_test, y_test = f['x_test'], f['y_test']
                return (x_train, y_train), (x_test, y_test)

        img_rows, img_cols = 28, 28
        num_classes = 10
        (x_train, y_train), (x_test, y_test) = _load_raw_dataset(raw_path)
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



class MNISTData(object):

    def __init__(self, data_path, batch_size, **kwargs):

        _, num_classes, X_train, y_train, X_val, y_val = load_dataset(data_path)
        
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

        self.num_classes = num_classes

        self.batch_size=batch_size
        
    def get_feature_shape(self):
        return self.X_train[0].shape
    
    def get_training_data(self):
        return self.X_train, self.y_train
    
    def get_validation_data(self):
        return self.X_val, self.y_val

    def get_training_data_size(self):
        return self.X_train.shape[0]

    def get_validation_data_size(self):
        return self.X_val.shape[0]

    @staticmethod
    def batch_generator(X, y, idxs, batch_size, num_batches):
        for i in range(num_batches):
            a = i * batch_size
            b = a + batch_size
            yield X[idxs[a:b]], y[idxs[a:b]]

    def get_batch_generator(self, train_or_val, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size

        if train_or_val == 'train':
            X = self.X_train
            y = self.y_train
        elif train_or_val == 'val':
            X = self.X_val
            y = self.y_val
        else:
            raise ValueError('dtype needs to be train or val')

        # shuffle data indices
        idxs = np.random.permutation(np.arange(X.shape[0]))

        # compute the number of batches
        num_batches = ceil(X.shape[0] / batch_size)

        # build the generator and return it
        return self.batch_generator(X, y, idxs, batch_size, num_batches)
        