import os
from math import ceil

import numpy as np
import tensorflow as tf
# from tqdm import tqdm

from ...datasets import load_dataset
from .tensorflowflutils import tf_get_vars, tf_get_tensor_dict, tf_set_tensor_dict


class TensorFlow2DUNet(object):

    def __init__(self, X_train, y_train, X_val, y_val):
        self.assign_ops = None
        self.placeholders = None

        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

        self.create_model()

    def create_model(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        # FIXME: shape should be derived from input data
        self.X = tf.placeholder(tf.float32, (None, 128, 128, 1))
        self.y = tf.placeholder(tf.float32, (None, 128, 128, 1))
        self.output = define_model(self.X, use_upsampling=True)

        self.loss = dice_coef_loss(self.y, self.output, smooth=32.0)
        self.validation_metric = dice_coef(self.y, self.output, smooth=1.0)

        self.global_step = tf.train.get_or_create_global_step()

        self.tvars = tf.trainable_variables()

        # self.optimizer = tf.train.AdamOptimizer(1e-4, beta1=0.9, beta2=0.999)
        self.optimizer = tf.train.RMSPropOptimizer(1e-5)

        self.gvs = self.optimizer.compute_gradients(self.loss, self.tvars)
        self.train_step = self.optimizer.apply_gradients(self.gvs,
                                                         global_step=self.global_step)
        
        self.opt_vars = self.optimizer.variables()

        # FIXME: Do we really need to share the opt_vars? Two opt_vars for one tvar: gradient and moment.
        self.fl_vars = self.tvars + self.opt_vars

        self.sess.run(tf.global_variables_initializer())

    def get_tensor_dict(self, with_opt_vars=True):
        if with_opt_vars is True:
            return tf_get_tensor_dict(self.sess, self.fl_vars)
        else:
            return tf_get_tensor_dict(self.sess, self.tvars)

    def set_tensor_dict(self, tensor_dict):
        self.assign_ops, self.placeholders = tf_set_tensor_dict(tensor_dict, self.sess, self.fl_vars, self.assign_ops, self.placeholders)

    def reset_opt_vars(self):
        # We may save the intialized values in the beginning and restore when needed here. We will waste some storage if most of them are actually 0s or 1s.
        # Instead, we just rerun the initializer of each variable.
        for var in self.opt_vars:
            if hasattr(var, 'initializer'):
                var.initializer.run(session=self.sess)
            else:
                self.logger.error("Failed to reset opt_var %s." % var.name)

    def train_epoch(self, epoch=None, batch_size=64):
        tf.keras.backend.set_learning_phase(True)

        # shuffle data
        idx = np.random.permutation(np.arange(self.X_train.shape[0]))
        X = self.X_train
        y = self.y_train

        # compute the number of batches
        num_batches = ceil(X.shape[0] / batch_size)

        losses = []
        # for i in tqdm(range(num_batches), desc="training epoch"):
        for i in range(num_batches):
            a = i * batch_size
            b = a + batch_size
            losses.append(self.train_batch(X[idx[a:b]], y[idx[a:b]]))

        return np.mean(losses)

    def validate(self, batch_size=64):
        tf.keras.backend.set_learning_phase(False)

        score = 0
        # for i in tqdm(np.arange(0, self.X_val.shape[0], batch_size), desc="validating"):
        for i in np.arange(0, self.X_val.shape[0], batch_size):
            X = self.X_val[i:i+batch_size]
            y = self.y_val[i:i+batch_size]
            weight = X.shape[0] / self.X_val.shape[0]
            _, s = self.validate_batch(X, y)
            score += s * weight
        return score

    def train_batch(self, X, y):
        feed_dict = {self.X: X, self.y: y}
        
        # run the train step and return the loss
        _, loss = self.sess.run([self.train_step, self.loss], feed_dict=feed_dict)
        return loss

    def validate_batch(self, X, y):
        feed_dict = {self.X: X, self.y: y}

        return self.sess.run([self.output, self.validation_metric], feed_dict=feed_dict)

    def get_training_data_size(self):
        return self.X_train.shape[0]

    def get_validation_data_size(self):
        return self.X_val.shape[0]


def dice_coef(y_true, y_pred, smooth=1.0, **kwargs):

    intersection = tf.reduce_sum(y_true * y_pred, axis=[1,2,3])
    coef = (tf.constant(2.) * intersection + tf.constant(smooth)) / \
           (tf.reduce_sum(y_true, axis=[1,2,3]) + tf.reduce_sum(y_pred, axis=(1,2,3)) + tf.constant(smooth))
    return tf.reduce_mean(coef)


def dice_coef_loss(y_true, y_pred, smooth=1.0, **kwargs):

    intersection = tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3))

    term1 = -tf.log(tf.constant(2.0) * intersection + smooth)
    term2 = tf.log(tf.reduce_sum(y_true, axis=(1, 2, 3)) +
                   tf.reduce_sum(y_pred, axis=(1, 2, 3)) + smooth)
    
    term1 = tf.reduce_mean(term1)
    term2 = tf.reduce_mean(term2)
    
    loss = term1 + term2

    return loss


CHANNEL_LAST = True
if CHANNEL_LAST:
    concat_axis = -1
    data_format = 'channels_last'
else:
    concat_axis = 1
    data_format = 'channels_first'

tf.keras.backend.set_image_data_format(data_format)


def define_model(input_tensor, use_upsampling=False, n_cl_out=1, dropout=0.2, print_summary = False, activation_function='relu', seed=0xFEEDFACE, **kwargs):
    # Set keras learning phase to train
    tf.keras.backend.set_learning_phase(True)

    # Don't initialize variables on the fly
    tf.keras.backend.manual_variable_initialization(False)

    inputs = tf.keras.layers.Input(tensor=input_tensor, name='Images')

    if activation_function == 'relu':
        activation = tf.nn.relu
    elif activation_function == 'leakyrelu':
        activation = tf.nn.leaky_relu
            
    params = dict(kernel_size=(3, 3), activation=activation,
                  padding='same', data_format=data_format,
                  # kernel_initializer='he_uniform')
                  kernel_initializer=tf.keras.initializers.he_uniform(seed=seed))
#                   kernel_initializer=tf.keras.initializers.he_normal(seed=0xFEEDFEACE))
#                   tf.keras.initializers.RandomUniform(minval=0.01, maxval=0.5, seed=0xFEEDFACE))
    
    conv1 = tf.keras.layers.Conv2D(name='conv1a', filters=32, **params)(inputs)
    conv1 = tf.keras.layers.Conv2D(name='conv1b', filters=32, **params)(conv1)
    pool1 = tf.keras.layers.MaxPooling2D(name='pool1', pool_size=(2, 2))(conv1)

    conv2 = tf.keras.layers.Conv2D(name='conv2a', filters=64, **params)(pool1)
    conv2 = tf.keras.layers.Conv2D(name='conv2b', filters=64, **params)(conv2)
    pool2 = tf.keras.layers.MaxPooling2D(name='pool2', pool_size=(2, 2))(conv2)

    conv3 = tf.keras.layers.Conv2D(name='conv3a', filters=128, **params)(pool2)
    conv3 = tf.keras.layers.Dropout(dropout)(conv3) ### Trying dropout layers earlier on, as indicated in the paper
    conv3 = tf.keras.layers.Conv2D(name='conv3b', filters=128, **params)(conv3)

    pool3 = tf.keras.layers.MaxPooling2D(name='pool3', pool_size=(2, 2))(conv3)

    conv4 = tf.keras.layers.Conv2D(name='conv4a', filters=256, **params)(pool3)
    conv4 = tf.keras.layers.Dropout(dropout)(conv4) ### Trying dropout layers earlier on, as indicated in the paper
    conv4 = tf.keras.layers.Conv2D(name='conv4b', filters=256, **params)(conv4)

    pool4 = tf.keras.layers.MaxPooling2D(name='pool4', pool_size=(2, 2))(conv4)

    conv5 = tf.keras.layers.Conv2D(name='conv5a', filters=512, **params)(pool4)
    conv5 = tf.keras.layers.Conv2D(name='conv5b', filters=512, **params)(conv5)

    if use_upsampling:
        up6 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling2D(name='up6', size=(2, 2))(conv5), conv4], axis=concat_axis)
    else:
        up6 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(name='transConv6', filters=256, data_format=data_format,
                           kernel_size=(2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=concat_axis)

    conv6 = tf.keras.layers.Conv2D(name='conv6a', filters=256, **params)(up6)
    conv6 = tf.keras.layers.Conv2D(name='conv6b', filters=256, **params)(conv6)

    if use_upsampling:
        up7 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling2D(name='up7', size=(2, 2))(conv6), conv3], axis=concat_axis)
    else:
        up7 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(name='transConv7', filters=128, data_format=data_format,
                           kernel_size=(2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=concat_axis)

    conv7 = tf.keras.layers.Conv2D(name='conv7a', filters=128, **params)(up7)
    conv7 = tf.keras.layers.Conv2D(name='conv7b', filters=128, **params)(conv7)

    if use_upsampling:
        up8 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling2D(name='up8', size=(2, 2))(conv7), conv2], axis=concat_axis)
    else:
        up8 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(name='transConv8', filters=64, data_format=data_format,
                           kernel_size=(2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=concat_axis)


    conv8 = tf.keras.layers.Conv2D(name='conv8a', filters=64, **params)(up8)
    conv8 = tf.keras.layers.Conv2D(name='conv8b', filters=64, **params)(conv8)

    if use_upsampling:
        up9 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling2D(name='up9', size=(2, 2))(conv8), conv1], axis=concat_axis)
    else:
        up9 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(name='transConv9', filters=32, data_format=data_format,
                           kernel_size=(2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=concat_axis)


    conv9 = tf.keras.layers.Conv2D(name='conv9a', filters=32, **params)(up9)
    conv9 = tf.keras.layers.Conv2D(name='conv9b', filters=32, **params)(conv9)

    conv10 = tf.keras.layers.Conv2D(name='Mask', filters=n_cl_out, kernel_size=(1, 1),
                    data_format=data_format, activation='sigmoid')(conv9)

    model = tf.keras.models.Model(inputs=[inputs], outputs=[conv10])

    if print_summary:
        print (model.summary())

    return conv10