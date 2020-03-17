# based on https://github.com/tensorflow/models/tree/master/resnet
import logging                                                                                                                                                                          
                                                                                                                                                                                        
import tensorflow as tf                                                                                                                                                                 
import tensorflow.keras as keras                                                                                                                                                        
from tensorflow.keras import backend as K                                                                                                                                               
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense                                                                                                                              
from .base import FLKerasModel                                                                                                                                                          
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import AveragePooling2D, Input, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
import numpy as np
import os

# Training parameters
batch_size = 32  # orig paper trained all networks with batch_size=128
epochs = 200
data_augmentation = True
num_classes = 10

import numpy as np
def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x

def resnet_v1(input_shape, depth, num_classes=10):
    """ResNet Version 1 Model builder [a]

    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model

def resnet_v2(input_shape, depth, num_classes=10):
    """ResNet Version 2 Model builder [b]

    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2    # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model

class ConvModel(FLKerasModel):
    """ResNet model."""
    def __init__(self, data, **kwargs):
        """ResNet constructor.

        Args:
          mode: One of 'train' and 'eval'.
        """
        super(ConvModel, self).__init__(data=data)                                                                                                       
        #TODO: set the mode
        mode = 'train'
        self.mode = mode
        self.logger = logging.getLogger(__name__)                                                                                                        
        self.model = self.build_model(data.get_feature_shape(), data.num_classes)                                                                        
        print(self.model.summary())
        if self.data.y_train is not None and self.data.y_val is not None:                                                                                
            print("Training set size: %d; Validation set size: %d" % (len(self.data.y_train), len(self.data.y_val)))                                     
                                                                                                                                                         
        self.is_initial = True                                                                                                                           
                                                                                                                                                         
        self.initial_opt_weights = self._get_weights_dict(self.model.optimizer)  

    @staticmethod
    def load_dataset():
        """
        Load the CIFAR10 dataset
        """
        img_rows, img_cols, img_channel = 32, 32, 3
        num_classes = 10                                                                                                                                 
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        if K.image_data_format() == 'channels_first':                                                                                                    
            x_train = x_train.reshape(x_train.shape[0], img_channel, img_rows, img_cols)                                                                           
            x_test = x_test.reshape(x_test.shape[0], img_channel, img_rows, img_cols)                                                                              
            input_shape = (img_channel, img_rows, img_cols)                                                                                                        
        else:                                                                                                                                            
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, img_channel)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, img_channel)
            input_shape = (img_rows, img_cols, img_channel)
                                                                                                                                                         
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

    def get_data_shard_idx(self, is_iid, splits, split_idx):
        """
        """
        if not(len(splits) > split_idx):
            self.logger.exception("Assertion failed: len(splits) > split_idx")
        if is_iid:
            pass
        else:
            raise NotImplementedError

    @staticmethod
    def build_model(input_shape, num_classes, depth=20):
        model = resnet_v1(input_shape=input_shape, depth=depth)
        #model = resnet_v2(input_shape=input_shape, depth=depth)
        model.compile(loss=keras.losses.categorical_crossentropy,                                                                                        
                        optimizer=keras.optimizers.Adam(),                                                                                               
                        metrics=['accuracy'])                                                                                                            
        return model
    
    def build_model_tmp(input_shape, num_classes):
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
        #return self._build_model()

  #########################################################################
    def add_internal_summaries(self):
        pass

    def _stride_arr(self, stride):
        """Map a stride scalar to the stride array for tf.nn.conv2d."""
        return [1, stride, stride, 1]

    def _build_model(self):
        assert self.mode == 'train' or self.mode == 'eval'
        """Build the core model within the graph."""
        with tf.variable_scope('input'):

          self.x_input = tf.placeholder(
            tf.float32,
            shape=[None, 32, 32, 3])

          self.y_input = tf.placeholder(tf.int64, shape=None)


          input_standardized = tf.map_fn(lambda img: tf.image.per_image_standardization(img),
                                   self.x_input)
          x = self._conv('init_conv', input_standardized, 3, 3, 16, self._stride_arr(1))



        strides = [1, 2, 2]
        activate_before_residual = [True, False, False]
        res_func = self._residual

        # Uncomment the following codes to use w28-10 wide residual network.
        # It is more memory efficient than very deep residual network and has
        # comparably good performance.
        # https://arxiv.org/pdf/1605.07146v1.pdf
        filters = [16, 160, 320, 640]


        # Update hps.num_residual_units to 9

        with tf.variable_scope('unit_1_0'):
          x = res_func(x, filters[0], filters[1], self._stride_arr(strides[0]),
                       activate_before_residual[0])
        for i in range(1, 5):
          with tf.variable_scope('unit_1_%d' % i):
            x = res_func(x, filters[1], filters[1], self._stride_arr(1), False)

        with tf.variable_scope('unit_2_0'):
          x = res_func(x, filters[1], filters[2], self._stride_arr(strides[1]),
                       activate_before_residual[1])
        for i in range(1, 5):
          with tf.variable_scope('unit_2_%d' % i):
            x = res_func(x, filters[2], filters[2], self._stride_arr(1), False)

        with tf.variable_scope('unit_3_0'):
          x = res_func(x, filters[2], filters[3], self._stride_arr(strides[2]),
                       activate_before_residual[2])
        for i in range(1, 5):
          with tf.variable_scope('unit_3_%d' % i):
            x = res_func(x, filters[3], filters[3], self._stride_arr(1), False)

        with tf.variable_scope('unit_last'):
          x = self._batch_norm('final_bn', x)
          x = self._relu(x, 0.1)
          x = self._global_avg_pool(x)

        with tf.variable_scope('logit'):
          self.pre_softmax = self._fully_connected(x, 10)

        self.predictions = tf.argmax(self.pre_softmax, 1)
        self.correct_prediction = tf.equal(self.predictions, self.y_input)
        self.num_correct = tf.reduce_sum(
            tf.cast(self.correct_prediction, tf.int64))
        self.accuracy = tf.reduce_mean(
            tf.cast(self.correct_prediction, tf.float32))

        with tf.variable_scope('costs'):
          self.y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
              logits=self.pre_softmax, labels=self.y_input)
          self.xent = tf.reduce_sum(self.y_xent, name='y_xent')
          self.mean_xent = tf.reduce_mean(self.y_xent)
          self.weight_decay_loss = self._decay()

    def _batch_norm(self, name, x):
        """Batch normalization."""
        with tf.name_scope(name):
          return tf.contrib.layers.batch_norm(
              inputs=x,
              decay=.9,
              center=True,
              scale=True,
              activation_fn=None,
              updates_collections=None,
              is_training=(self.mode == 'train'))

    def _residual(self, x, in_filter, out_filter, stride,
                    activate_before_residual=False):
        """Residual unit with 2 sub layers."""
        if activate_before_residual:
          with tf.variable_scope('shared_activation'):
            x = self._batch_norm('init_bn', x)
            x = self._relu(x, 0.1)
            orig_x = x
        else:
          with tf.variable_scope('residual_only_activation'):
            orig_x = x
            x = self._batch_norm('init_bn', x)
            x = self._relu(x, 0.1)

        with tf.variable_scope('sub1'):
          x = self._conv('conv1', x, 3, in_filter, out_filter, stride)

        with tf.variable_scope('sub2'):
          x = self._batch_norm('bn2', x)
          x = self._relu(x, 0.1)
          x = self._conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])

        with tf.variable_scope('sub_add'):
          if in_filter != out_filter:
            orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
            orig_x = tf.pad(
                orig_x, [[0, 0], [0, 0], [0, 0],
                         [(out_filter-in_filter)//2, (out_filter-in_filter)//2]])
          x += orig_x

        tf.logging.debug('image after unit %s', x.get_shape())
        return x

    def _decay(self):
        """L2 weight decay loss."""
        costs = []
        for var in tf.trainable_variables():
          if var.op.name.find('DW') > 0:
            costs.append(tf.nn.l2_loss(var))
        return tf.add_n(costs)

    def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
        """Convolution."""
        with tf.variable_scope(name):
          n = filter_size * filter_size * out_filters
          kernel = tf.get_variable(
              'DW', [filter_size, filter_size, in_filters, out_filters],
              tf.float32, initializer=tf.random_normal_initializer(
                  stddev=np.sqrt(2.0/n)))
          return tf.nn.conv2d(x, kernel, strides, padding='SAME')

    def _relu(self, x, leakiness=0.0):
        """Relu, with optional leaky support."""
        return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

    def _fully_connected(self, x, out_dim):
        """FullyConnected layer for final output."""
        num_non_batch_dimensions = len(x.shape)
        prod_non_batch_dimensions = 1
        for ii in range(num_non_batch_dimensions - 1):
          prod_non_batch_dimensions *= int(x.shape[ii + 1])
        x = tf.reshape(x, [tf.shape(x)[0], -1])
        w = tf.get_variable(
            'DW', [prod_non_batch_dimensions, out_dim],
            initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
        b = tf.get_variable('biases', [out_dim],
                            initializer=tf.constant_initializer())
        return tf.nn.xw_plus_b(x, w, b)

    def _global_avg_pool(self, x):
        assert x.get_shape().ndims == 4
        return tf.reduce_mean(x, [1, 2])
