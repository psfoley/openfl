import tensorflow as tf
from math import ceil
import numpy as np
from tqdm import tqdm

from .tensorflowflutils import tf_get_vars, tf_get_tensor_dict, tf_set_tensor_dict, tf_reset_vars


class TensorFlow2DUNet(object):

    def __init__(self, X_train, y_train, X_val, y_val, **kwargs):
        self.assign_ops = None
        self.placeholders = None

        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

        self.create_model(**kwargs)

    def create_model(self, **kwargs):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.intra_op_parallelism_threads = 112
        config.inter_op_parallelism_threads = 1
        self.sess = tf.Session(config=config)

        # FIXME: shape should be derived from input data
        self.X = tf.placeholder(tf.float32, (None, 128, 128, 1))
        self.y = tf.placeholder(tf.float32, (None, 128, 128, 1))
        self.output = define_model(self.X, use_upsampling=True, **kwargs)

        self.loss = dice_coef_loss(self.y, self.output, smooth=32.0)
        self.validation_metric = dice_coef(self.y, self.output, smooth=1.0)

        self.global_step = tf.train.get_or_create_global_step()

        self.tvars = tf.trainable_variables()

        # self.optimizer = tf.train.AdamOptimizer(4e-5, beta1=0.9, beta2=0.999)
        self.optimizer = tf.train.RMSPropOptimizer(1e-5)

        self.gvs = self.optimizer.compute_gradients(self.loss, self.tvars)
        self.train_step = self.optimizer.apply_gradients(self.gvs,
                                                         global_step=self.global_step)
        
        self.opt_vars = self.optimizer.variables()

        # FIXME: Do we really need to share the opt_vars? Two opt_vars for one tvar: gradient and square sum for RMSprop.
        self.fl_vars = self.tvars + self.opt_vars

        self.sess.run(tf.global_variables_initializer())

    def get_tensor_dict(self, with_opt_vars=True):
        if with_opt_vars is True:
            return tf_get_tensor_dict(self.sess, self.fl_vars)
        else:
            return tf_get_tensor_dict(self.sess, self.tvars)

    def set_tensor_dict(self, tensor_dict, with_opt_vars):
        """Set the tensor dictionary (weights) with new values.

        Parameters
        ----------
        tensor_dict : dict
            Weights.
        with_opt_vars : bool
            If we should set the variablies of the optimizer. (TODO: not supported yet.)
        """
        self.assign_ops, self.placeholders = tf_set_tensor_dict(tensor_dict, self.sess, self.fl_vars, self.assign_ops, self.placeholders)

    def reset_opt_vars(self):
        return tf_reset_vars(self.sess, self.opt_vars)

    def train_epoch(self, epoch=None, batch_size=64, use_tqdm=False):
        tf.keras.backend.set_learning_phase(True)

        # shuffle data
        idx = np.random.permutation(np.arange(self.X_train.shape[0]))
        X = self.X_train
        y = self.y_train

        # compute the number of batches
        num_batches = ceil(X.shape[0] / batch_size)

        losses = []
        if use_tqdm:
            gen = tqdm(range(num_batches), desc="training epoch")
        else:
            gen = range(num_batches)
        for i in gen:
            a = i * batch_size
            b = a + batch_size
            losses.append(self.train_batch(X[idx[a:b]], y[idx[a:b]]))

        return np.mean(losses)

    def validate(self, batch_size=64, use_tqdm=False):
        tf.keras.backend.set_learning_phase(False)

        score = 0
        if use_tqdm:
            gen = tqdm(np.arange(0, self.X_val.shape[0], batch_size), desc="validating")
        else:
            gen = np.arange(0, self.X_val.shape[0], batch_size)
        for i in gen:
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


def define_model(input_tensor,
                 use_upsampling=False,
                 n_cl_out=1,
                 dropout=0.2,
                 print_summary = False,
                 activation_function='relu',
                 seed=0xFEEDFACE,
                 depth=5,
                 dropout_at=[2,3],
                 initial_filters=32,
                 batch_norm=True,
                 **kwargs):

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
                  kernel_initializer=tf.keras.initializers.he_uniform(seed=seed))
    
    convb_layers = {}

    net = inputs
    filters = initial_filters
    for i in range(depth):
        name = 'conv{}a'.format(i+1)
        net = tf.keras.layers.Conv2D(name=name, filters=filters, **params)(net)
        if i in dropout_at:
            net = tf.keras.layers.Dropout(dropout)(net)
        name = 'conv{}b'.format(i+1)
        net = tf.keras.layers.Conv2D(name=name, filters=filters, **params)(net)
        if batch_norm:
            net = tf.keras.layers.BatchNormalization()(net)
        convb_layers[name] = net
        # only pool if not last level
        if i != depth - 1:
            name = 'pool{}'.format(i+1)
            net = tf.keras.layers.MaxPooling2D(name=name, pool_size=(2, 2))(net)
            filters *= 2

    # do the up levels
    filters //= 2
    for i in range(depth - 1):        
        if use_upsampling:
            up = tf.keras.layers.UpSampling2D(name='up{}'.format(depth + i + 1), size=(2, 2))(net) 
        else:
            up = tf.keras.layers.Conv2DTranspose(name='transConv6', filters=filters, data_format=data_format, kernel_size=(2, 2), strides=(2, 2), padding='same')(net)
        net = tf.keras.layers.concatenate([up, convb_layers['conv{}b'.format(depth - i - 1)]], axis=concat_axis)
        net = tf.keras.layers.Conv2D(name='conv{}a'.format(depth + i + 1), filters=filters, **params)(net)
        net = tf.keras.layers.Conv2D(name='conv{}b'.format(depth + i + 1), filters=filters, **params)(net)
        filters //= 2

    net = tf.keras.layers.Conv2D(name='Mask', filters=n_cl_out, kernel_size=(1, 1), data_format=data_format, activation='sigmoid')(net)

    model = tf.keras.models.Model(inputs=[inputs], outputs=[net])

    if print_summary:
        print (model.summary())

    return net
