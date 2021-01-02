# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf

def UpSampling2D(input):
    new_shape = tf.shape(input)[1:3]
    new_shape *= tf.constant(np.array([2, 2]).astype('int32'))
    output = tf.image.resize_nearest_neighbor(input, new_shape)
    return output


def Encoder_decoder(inputs, is_training):
    with tf.variable_scope('DIP'):
        # Encoder
        enconv1_64 = tf.layers.conv2d(inputs, 64, 3, (2, 2), padding='same', activation=tf.nn.relu)

        enconv2_128 = tf.layers.conv2d(enconv1_64, 128, 3, (2, 2), padding='same')
        enconv2_128 = tf.nn.relu(tf.layers.batch_normalization(enconv2_128, training=is_training))

        enconv3_256 = tf.layers.conv2d(enconv2_128, 256, 1, (2, 2), padding='same')
        enconv3_256 = tf.nn.relu(tf.layers.batch_normalization(enconv3_256, training=is_training))

        enconv4_512 = tf.layers.conv2d(enconv3_256, 512, 1, (1, 1), padding='same')
        enconv4_512 = tf.nn.relu(tf.layers.batch_normalization(enconv4_512, training=is_training))
        #
        # enconv5_512 = tf.layers.conv2d(enconv4_512, 512, 3,  (2, 2), padding='same')
        # enconv5_512 = tf.nn.relu(tf.layers.batch_normalization(enconv5_512, training=is_training))
        #
        # enconv6_512 = tf.layers.conv2d(enconv5_512, 512, 1, (2, 2), padding='same')
        # enconv6_512 = tf.nn.relu(tf.layers.batch_normalization(enconv6_512, training=is_training))

        #Decoder
        deconv1_512 = UpSampling2D(enconv4_512)
        deconv1_512 = tf.layers.conv2d(deconv1_512, 512, 1, strides=(1, 1), padding='same')
        deconv1_512 = tf.nn.relu(tf.layers.batch_normalization(deconv1_512, training=is_training))

        deconv2_512 = UpSampling2D(deconv1_512)
        deconv2_512 = tf.layers.conv2d(deconv2_512, 512, 3, strides=(1, 1), padding='same')
        deconv2_512 = tf.nn.relu(tf.layers.batch_normalization(deconv2_512, training=is_training))

        deconv3_256 = UpSampling2D(deconv2_512)
        deconv3_256 = tf.layers.conv2d(deconv3_256, 256, 3, strides=(1, 1), padding='same')
        deconv3_256 = tf.nn.relu(tf.layers.batch_normalization(deconv3_256, training=is_training))

        deconv4_128 = UpSampling2D(deconv3_256)
        deconv4_128 = tf.layers.conv2d(deconv4_128, 128, 5, strides=(1, 1), padding='same')
        deconv4_128 = tf.nn.relu(tf.layers.batch_normalization(deconv4_128, training=is_training))

        deconv5_64 = UpSampling2D(deconv4_128)
        deconv5_64 = tf.layers.conv2d(deconv5_64, 64, 5, strides=(1, 1), padding='same')
        deconv5_64 = tf.nn.relu(tf.layers.batch_normalization(deconv5_64, training=is_training))

        deconv6_3 = UpSampling2D(deconv5_64)
        deconv6_3 = tf.layers.conv2d(deconv6_3, 1, 5, strides=(1, 1), padding='same')
        output = tf.nn.sigmoid(deconv6_3)

        return output


def lossing(gen, gt):
    L_loss = tf.reduce_sum(tf.square(gen - gt), axis=[1,2,3])
    loss_G = tf.reduce_mean(L_loss)

    return loss_G

def optimization(loss):
    optimizer = tf.train.AdamOptimizer(0.001, name='AdamOptimizer')
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss)
    return train_op