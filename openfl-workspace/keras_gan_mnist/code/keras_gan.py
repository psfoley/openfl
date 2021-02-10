# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""You may copy this file as the starting point of your own model."""
from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow.keras import Model, layers
import numpy as np
from openfl.federated.data import FederatedDataLoader


# Create TF Model.
class Generator(Model):
    # Set layers.
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = layers.Dense(7 * 7 * 128)
        self.bn1 = layers.BatchNormalization()
        self.conv2tr1 = layers.Conv2DTranspose(64, 5, strides=2, padding='SAME')
        self.bn2 = layers.BatchNormalization()
        self.conv2tr2 = layers.Conv2DTranspose(1, 5, strides=2, padding='SAME')

    # Set forward pass.
    def call(self, x, is_training=False):
        x = self.fc1(x)
        x = self.bn1(x, training=is_training)
        x = tf.nn.leaky_relu(x)
        # Reshape to a 4-D array of images: (batch, height, width, channels)
        # New shape: (batch, 7, 7, 128)
        x = tf.reshape(x, shape=[-1, 7, 7, 128])
        # Deconvolution, image shape: (batch, 14, 14, 64)
        x = self.conv2tr1(x)
        x = self.bn2(x, training=is_training)
        x = tf.nn.leaky_relu(x)
        # Deconvolution, image shape: (batch, 28, 28, 1)
        x = self.conv2tr2(x)
        x = tf.nn.tanh(x)
        return x

# Generator Network
# Input: Noise, Output: Image
# Note that batch normalization has different behavior at training and inference time,
# we then use a placeholder to indicates the layer if we are training or not.
class Discriminator(Model):
    # Set layers.
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = layers.Conv2D(64, 5, strides=2, padding='SAME')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(128, 5, strides=2, padding='SAME')
        self.bn2 = layers.BatchNormalization()
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(1024)
        self.bn3 = layers.BatchNormalization()
        self.fc2 = layers.Dense(2)

    # Set forward pass.
    def call(self, x, is_training=False):
        x = tf.reshape(x, [-1, 28, 28, 1])
        x = self.conv1(x)
        x = self.bn1(x, training=is_training)
        x = tf.nn.leaky_relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=is_training)
        x = tf.nn.leaky_relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.bn3(x, training=is_training)
        x = tf.nn.leaky_relu(x)
        return self.fc2(x)


class KerasGAN:
    """A basic convolutional neural network model."""

    def __init__(self, **kwargs):
        """
        Initialize.

        Args:
            **kwargs: Additional parameters to pass to the function
        """

        x_train, y_train, x_test, y_test = self.setup_data()

        self.batch_size = 128
        self.training_steps = 100
        self.display_step = 10
        self.noise_dim = 100

        train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_data = train_data.repeat().shuffle(10000).batch(self.batch_size).prefetch(1)
        self.train_data = FederatedDataLoader(train_data)

        self.generator = Generator()
        self.discriminator = Discriminator()

        self.optimizer_gen = tf.optimizers.Adam(learning_rate=0.0002)
        self.optimizer_disc = tf.optimizers.Adam(learning_rate=0.0002)
        self.train(warmup=True)

    
    def train(self,warmup=False):
        # Run training for the given number of steps.
        gen_loss = 0.0
        disc_loss = 0.0
        for step, (batch_x, _) in enumerate(self.train_data.take(self.training_steps + 1)):
            
            if step == 0:
                # Generate noise.
                noise = np.random.normal(-1., 1., size=[self.batch_size, self.noise_dim]).astype(np.float32)
                gen_loss = self.generator_loss(self.discriminator(self.generator(noise)))
                disc_loss = self.discriminator_loss(self.discriminator(batch_x), self.discriminator(self.generator(noise)))
                print("initial: gen_loss: %f, disc_loss: %f" % (gen_loss, disc_loss))
                if warmup:
                    break
                continue
            
            # Run the optimization.
            gen_loss, disc_loss = self.run_optimization(batch_x)
            
            if step % self.display_step == 0:
                print("step: %i, gen_loss: %f, disc_loss: %f" % (step, gen_loss, disc_loss))

        disc_loss = float(disc_loss)
        gen_loss = float(gen_loss)
        return gen_loss, disc_loss

    # Losses.
    def generator_loss(self,reconstructed_image):
        gen_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=reconstructed_image, labels=tf.ones([self.batch_size], dtype=tf.int32)))
        return gen_loss
    
    def discriminator_loss(self,disc_fake, disc_real):
        disc_loss_real = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=disc_real, labels=tf.ones([self.batch_size], dtype=tf.int32)))
        disc_loss_fake = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=disc_fake, labels=tf.zeros([self.batch_size], dtype=tf.int32)))
        return disc_loss_real + disc_loss_fake

    # Optimization process. Inputs: real image and noise.
    def run_optimization(self,real_images):
        
        # Rescale to [-1, 1], the input range of the discriminator
        real_images = real_images * 2. - 1.
    
        # Generate noise.
        noise = np.random.normal(-1., 1., size=[self.batch_size, self.noise_dim]).astype(np.float32)
        
        with tf.GradientTape() as g:
                
            fake_images = self.generator(noise, is_training=True)
            disc_fake = self.discriminator(fake_images, is_training=True)
            disc_real = self.discriminator(real_images, is_training=True)
    
            disc_loss = self.discriminator_loss(disc_fake, disc_real)
                
        # Training Variables for each optimizer
        gradients_disc = g.gradient(disc_loss,  self.discriminator.trainable_variables)
        self.optimizer_disc.apply_gradients(zip(gradients_disc,  self.discriminator.trainable_variables))
        
        # Generate noise.
        noise = np.random.normal(-1., 1., size=[self.batch_size, self.noise_dim]).astype(np.float32)
        
        with tf.GradientTape() as g:
                
            fake_images = self.generator(noise, is_training=True)
            disc_fake = self.discriminator(fake_images, is_training=True)
    
            gen_loss = self.generator_loss(disc_fake)
                
        gradients_gen = g.gradient(gen_loss, self.generator.trainable_variables)
        self.optimizer_gen.apply_gradients(zip(gradients_gen, self.generator.trainable_variables))
        
        return gen_loss, disc_loss

    def setup_data(self):
        from tensorflow.keras.datasets import mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        # Convert to float32.
        x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
        # Normalize images value from [0, 255] to [0, 1].
        x_train, x_test = x_train / 255., x_test / 255.
        return x_train,y_train,x_test,y_test

if __name__ == '__main__':
    gan = KerasGAN()
    print(f'Length of train loader = {len(gan.train_data)}')
