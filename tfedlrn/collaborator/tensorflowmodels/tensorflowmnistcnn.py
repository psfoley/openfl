import os
from math import ceil

import numpy as np
import tensorflow as tf

from ...datasets import load_dataset
from .tensorflowflutils import tf_get_vars, tf_get_tensor_dict, tf_set_tensor_dict


class TensorFlowMNISTCNN(object):

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

        self.X = tf.placeholder(tf.float32, (None, 28, 28, 1))
        self.y = tf.placeholder(tf.float32, (None, 1))

        self.output = define_model(self.X)

        self.loss = tf.keras.metrics.categorical_crossentropy(self.y, self.output)
        self.validation_metric = tf.keras.metrics.categorical_accuracy(self.y, self.output)

        self.global_step = tf.train.get_or_create_global_step()

        self.tvars = tf.trainable_variables()

        self.optimizer = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.5)

        self.gvs = self.optimizer.compute_gradients(self.loss, self.tvars)
        self.train_step = self.optimizer.apply_gradients(self.gvs,
                                                         global_step=self.global_step)
        
        self.opt_vars = self.optimizer.variables()

        self.fl_vars = self.tvars + self.opt_vars

        self.sess.run(tf.global_variables_initializer())

    def get_tensor_dict(self):
        return tf_get_tensor_dict(self.sess, self.fl_vars)

    def set_tensor_dict(self, tensor_dict):
        self.assign_ops, self.placeholders = \
            tf_set_tensor_dict(tensor_dict, self.sess, self.fl_vars, self.assign_ops, self.placeholders)

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




def define_model(input_tensor, print_summary = False):
    # Set keras learning phase to train
    tf.keras.backend.set_learning_phase(True)

    # Don't initialize variables on the fly
    tf.keras.backend.manual_variable_initialization(False)

    inputs = tf.keras.layers.Input(tensor=input_tensor, name='Images')
            
    conv1 = tf.keras.layers.Conv2D(name='conv1', filters=20, kernel_size=(5, 5), 
                                   activation='relu')(inputs)
    pool1 = tf.keras.layers.MaxPooling2D(name='pool1', pool_size=(2, 2))(conv1)
    
    conv2 = tf.keras.layers.Conv2D(name='conv2', filters=50, kernel_size=(5, 5), 
                                   activation='relu')(pool1)
    pool2 = tf.keras.layers.MaxPooling2D(name='pool2', pool_size=(2, 2))(conv2)

    flatten = tf.keras.layers.Flatten(name='flatten')(pool2)

    fc1 = tf.keras.layers.Dense(name='fc1', units=500, activation='relu')(flatten)

    fc2 = tf.keras.layers.Dense(name='fc2', units=10, activation='softmax')(fc1)
    
    model = tf.keras.models.Model(inputs=[inputs], outputs=[fc2])

    if print_summary:
        print (model.summary())

    return fc2
