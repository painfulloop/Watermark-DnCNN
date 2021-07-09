# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf


def dncnn(input, is_training):
    with tf.variable_scope('block1', reuse=tf.AUTO_REUSE):
        output = tf.layers.conv2d(input, 64, 3, padding='same', activation=tf.nn.relu)
    for layers in range(2, 16 + 1):
        with tf.variable_scope('block%d' % layers, reuse=tf.AUTO_REUSE):
            output = tf.layers.conv2d(output, 64, 3, padding='same', name='conv%d' % layers, use_bias=False)
            output = tf.nn.relu(tf.layers.batch_normalization(output, training=is_training))
    with tf.variable_scope('block17', reuse=tf.AUTO_REUSE):
        output = tf.layers.conv2d(output, 1, 3, padding='same')
    return input - output, output


def lossing(Y, GT, batch_size):
    loss = (1.0 / batch_size) * tf.nn.l2_loss(Y - GT)
    return loss


def optimizer(loss, lr):
    optimizer = tf.train.AdamOptimizer(lr, name='AdamOptimizer')
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss)
    return train_op


if __name__ == '__main__':
    with tf.Graph().as_default():
        data = np.random.normal(size=(3, 3, 16))
        data = np.expand_dims(data, axis=0)
        input1 = tf.placeholder(tf.float32, shape=[1, 3, 3, 16])
        output = dncnn(input1, is_training=True)
        init = tf.global_variables_initializer()
        var_list = [t.name for t in tf.all_variables() if t.name.startswith('block')]
        op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        ops = [t for t in tf.GraphKeys.UPDATE_OPS ]
        print (var_list)
        print (op)
        print(ops)
        with tf.Session() as sess:
            sess.run(init)
            # out = sess.run(ops)
            # print (out)