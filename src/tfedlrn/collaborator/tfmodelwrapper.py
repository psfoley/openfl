import os
from math import ceil

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from .aipgunet import define_model, dice_coef, dice_coef_loss

class TFModelWrapper(ModelWrapper):
    """WIP code. Goal is to simplify porting a model to this framework.
    Currently, this creates a placeholder and assign op for every variable, which grows the graph considerably.
    Also, the abstraction for the tf.session isn't ideal yet."""

    def __init__(self, tfmodel_fn, session_config=None):
        self.sess = tf.Session(config=self.session_config)

        self.tfmodel = tfmodel_fn(self.sess)

        self.vars = tfmodel.get_vars()

    def get_tensor_dict(self):
        """FIXME: how to protect against redundant graphs?"""
        return {v.name: self.sess.run(v) for v in self.vars}

    def set_tensors_from_dict(self, tensor_dict):
        """FIXME: how to protect against redundant graphs?"""
        for k, v in tensor_dict:
            sess.run(self.assign_ops[k], feed_dict={self.placeholders[k]:v})

    def train_epoch(self):
        self.tfmodel.init_epoch()

        feed_dict = self.tfmodel.get_training_batch_feed_dict()
        
        # run the train step and return the loss
        _, loss = self.sess.run([self.train_step, self.loss], feed_dict=feed_dict)
        return loss

        pass

    @abc.abstractmethod
    def get_training_data_size(self):
        pass

    @abc.abstractmethod
    def validate(self):
        pass

    @abc.abstractmethod
    def get_validation_data_size(self):
        pass


    def get_model_parameters(self):
        raise NotImplementedError()

    def get_optimizer_parameters(self):
        raise NotImplementedError()

    def get_training_data_size(self):
        raise NotImplementedError()

    def get_validation_data_size(self):
        raise NotImplementedError()

    def set_model_parameters(self, params):
        raise NotImplementedError()

    def set_optimizer_parameters(self, params):
        raise NotImplementedError()



    @staticmethod
    def _placeholders_and_assign_ops_for_vars(vars):
        placeholders = [tf.placeholder(v.dtype, shape=v.shape) for v in vars]
        assign_ops = [tf.assign(v, p) for v, p in zip(vars, placeholders)]
        return placeholders, assign_ops

    def init_assign_ops(self):
        assert self.tvars is not None

        p, a = self._placeholders_and_assign_ops_for_vars(self.tvars)
        self.placeholders = p
        self.assign_ops = a

        if self.opt_vars is not None:
            p, a = self._placeholders_and_assign_ops_for_vars(self.opt_vars)
            self.opt_placeholders = p
            self.opt_assign_ops = a


    def __init__(self, **kwargs):
        self.load_data(data_path, load_mutiple)
        self.create_model(**kwargs)

    # FIXME: better abstraction to address this test case
    def load_data(self, data_path, load_mutiple=None):
        if load_mutiple is None:
            self.X_train = np.load(os.path.join(data_path, 'imgs_train.npy'))
            self.y_train = np.load(os.path.join(data_path, 'msks_train.npy'))
            self.X_val = np.load(os.path.join(data_path, 'imgs_val.npy'))
            self.y_val = np.load(os.path.join(data_path, 'msks_val.npy'))
        else:
            self.X_train = np.concatenate([np.load(os.path.join(data_path, str(i), 'imgs_train.npy')) for i in load_mutiple])
            self.y_train = np.concatenate([np.load(os.path.join(data_path, str(i), 'msks_train.npy')) for i in load_mutiple])
            self.X_val = np.concatenate([np.load(os.path.join(data_path, str(i), 'imgs_val.npy')) for i in load_mutiple])
            self.y_val = np.concatenate([np.load(os.path.join(data_path, str(i), 'msks_val.npy')) for i in load_mutiple])
            print('test data size:', self.X_train.shape[0] / 155, 'patients')

    def create_model(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        self.X = tf.placeholder(tf.float32, (None, 128, 128, 1))
        self.y = tf.placeholder(tf.float32, (None, 128, 128, 1))
        self.output = define_model(self.X, use_upsampling=True)

        self.loss = dice_coef_loss(self.y, self.output, smooth=32.0)
        self.validation_metric = dice_coef(self.y, self.output, smooth=1.0)

        self.global_step = tf.train.get_or_create_global_step()

        self.tvars = tf.trainable_variables()

        self.optimizer = tf.train.AdamOptimizer(1e-4, beta1=0.9, beta2=0.999)

        self.gvs = self.optimizer.compute_gradients(self.loss, self.tvars)
        self.train_step = self.optimizer.apply_gradients(self.gvs,
                                                         global_step=self.global_step)
        
        self.opt_vars = self.optimizer.variables()

        self.sess.run(tf.global_variables_initializer())

    def train_epoch(self, batch_size=64):
        # shuffle data
        idx = np.random.permutation(np.arange(self.X_train.shape[0]))
        X = self.X_train[idx]
        y = self.y_train[idx]

        # compute the number of batches
        num_batches = ceil(X.shape[0] / batch_size)

        losses = []
        for i in tqdm(range(num_batches), desc="training epoch"):
            a = i * batch_size
            b = a + batch_size
            losses.append(self.train_batch(X[a:b], y[a:b]))

        return np.mean(losses)

    def validate(self, batch_size=64):
        score = 0
        for i in tqdm(np.arange(0, self.X_val.shape[0], batch_size), desc="validating"):
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

    # FIXME: remove dropout from validation
    def validate_batch(self, X, y):
        feed_dict = {self.X: X, self.y: y}

        return self.sess.run([self.output, self.validation_metric], feed_dict=feed_dict)
