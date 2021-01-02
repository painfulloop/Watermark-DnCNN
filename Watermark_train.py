# -*- coding: utf-8 -*-
import os, cv2
import numpy as np
import DnCNNModel
import tensorflow as tf

train_data = './data/img_clean_pats.npy'
org_model_path = './DnCNN_weight/' #folder containing weights of original DnCNN

overwriting_path = './overwriting/' #folder containing new weights created in this script ( model trained with trigger key)

np.random.seed(0)

lambda_ = 0.001
image_mod = 0
type = 'cman'
spec_size = [1, 40, 40, 1]

DnCNN_model_name = 'Black_DnCNN_' + type + '_weight_'


# def transition(w, s_x, s_y):
#     filtered_x = tf.nn.conv2d(w, s_x, strides=[1, 1, 1, 1], padding='SAME')
#     filtered_y = tf.nn.conv2d(w, s_y, strides=[1, 1, 1, 1], padding='SAME')
#     filtered = filtered_x + filtered_y
#     filtered = tf.reshape(filtered, [special_num, np.prod(filtered.get_shape().as_list())])
#     return filtered

def transition(w):
    return w


def watermark_loss(gen, gt):
    # gt = tf.reshape(gt, [special_num, np.prod(gt.get_shape().as_list())])
    # loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=gt, logits=gen)
    # loss = tf.reduce_mean(loss)
    loss = tf.reduce_sum(tf.square(gen - gt), axis=[1, 2, 3])
    loss = tf.reduce_mean(loss)
    return loss


def ft_DnCNN_optimizer(dncnn_loss, line_loss, lr):
    loss = dncnn_loss + lambda_ * line_loss

    optimizer = tf.train.AdamOptimizer(lr, name='AdamOptimizer')
    ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(ops):
        var_list = [t for t in tf.trainable_variables()]
        gradient = optimizer.compute_gradients(loss, var_list=var_list)
        train_op = optimizer.apply_gradients(gradient)
    return train_op


def sobel():
    a = np.array([1, 4, 5, 0, -5, -4, -1])
    b = np.array([1, 6, 15, 20, 15, 6, 1])
    a = np.reshape(a, [1, a.shape[0]])
    b = np.reshape(b, [1, b.shape[0]])
    c = np.matmul(b.transpose(), a)
    sobel_base = tf.constant(c, tf.float32)
    sobel_x_filter = tf.reshape(sobel_base, [7, 7, 1, 1])
    sobel_y_filter = tf.transpose(sobel_x_filter, [1, 0, 2, 3])
    return sobel_x_filter, sobel_y_filter


def train(epochs=8, batch_size=128,learn_rate=0.0001, sigma=25):
    special_num = 5
    with tf.Graph().as_default():
        lr = tf.placeholder(tf.float32, shape=[], name='learning_rate')
        training = tf.placeholder(tf.bool, name='is_training')
        img_clean = tf.placeholder(tf.float32, [batch_size, spec_size[1], spec_size[2], spec_size[3]],
                                   name='clean_image')
        img_spec = tf.placeholder(tf.float32, [special_num, spec_size[1], spec_size[2], spec_size[3]],
                                  name='spec_image')
        special_gt = tf.placeholder(tf.float32, [special_num, spec_size[1], spec_size[2], spec_size[3]])

        # DnCNN model
        img_noise = img_clean + tf.random_normal(shape=tf.shape(img_clean), stddev=sigma / 255.0) #dati con aggiunta di rumore
        img_total = tf.concat([img_noise, img_spec], 0)   # concatenazione img_noise e img trigger
        Y, N = DnCNNModel.dncnn(img_total, is_training=training)

        # slide
        Y_img = tf.slice(Y, [0, 0, 0, 0], [batch_size, spec_size[1], spec_size[2], spec_size[3]])
        N_spe = tf.slice(N, [batch_size, 0, 0, 0], [special_num, spec_size[1], spec_size[2], spec_size[3]])

        # host loss
        dncnn_loss = DnCNNModel.lossing(Y_img, img_clean, batch_size)

        # sobel_x, sobel_y = sobel()
        # extract weight
        dncnn_s_out = transition(N_spe)

        # mark loss
        mark_loss = watermark_loss(dncnn_s_out, special_gt) #special_gt = verification img

        # Update model
        dncnn_opt = ft_DnCNN_optimizer(dncnn_loss, mark_loss, lr)

        init = tf.global_variables_initializer()

        dncnn_var_list = [v for v in tf.global_variables() if v.name.startswith('block')]
        DnCNN_saver = tf.train.Saver(dncnn_var_list, max_to_keep=50)

        with tf.Session() as sess:
            data_total = np.load(train_data)
            data_total = data_total.astype(np.float32) / 255.0
            num_example, row, col, chanel = data_total.shape
            numBatch = num_example // batch_size

            # special_input = cv2.imread('./input_data/spec_input.png', 0)  #trigger img
            special_input = cv2.imread('key_imgs/trigger_image.png', 0)
            special_input = special_input.astype(np.float32) / 255.0
            special_input = np.expand_dims(special_input, 0)
            special_input = np.expand_dims(special_input, 3)

            special_input = np.repeat(special_input, special_num, axis=0)

            # daub_Images = cv2.imread('./input_data/spec_gt.png', 0) #verification img
            daub_Images = cv2.imread('key_imgs/verification_image.png', 0)
            daub_Images = daub_Images.astype(np.float32) / 255.0
            daub_Images = np.expand_dims(daub_Images, 0)
            daub_Images = np.expand_dims(daub_Images, 3)

            daub_Images = np.repeat(daub_Images, special_num, axis=0)

            sess.run(init)

            ckpt = tf.train.get_checkpoint_state(org_model_path)
            if ckpt and ckpt.model_checkpoint_path:
                full_path = tf.train.latest_checkpoint(org_model_path)
                print(full_path)
                DnCNN_saver.restore(sess, full_path)
                print("Loading " + os.path.basename(full_path) + " to the model")

            else:
                print("DnCNN weight must be exist")
                assert ckpt != None, 'weights not exist'

            step = 0
            for epoch in range(0, epochs):
                np.random.shuffle(data_total)
                for batch_id in range(0, numBatch):

                    # tag = np.random.randint(0, 26)
                    special_input = special_input + 0 * np.random.normal(size=special_input.shape) / 255
                    batch_images = data_total[batch_id * batch_size:(batch_id + 1) * batch_size, :, :, :]

                    if batch_id % 100 == 0:
                        dncnn_lost = sess.run(dncnn_loss, feed_dict={img_clean: batch_images, lr: learn_rate,
                                                                     img_spec: special_input, training: False})
                        mark_lost = sess.run(mark_loss, feed_dict={img_clean: batch_images, img_spec: special_input,
                                                                   lr: learn_rate,
                                                                   special_gt: daub_Images, training: False})

                        print("step = %d, dncnn_loss = %f,mark_loss = %f" % (step, dncnn_lost, mark_lost))

                    _ = sess.run(dncnn_opt, feed_dict={img_clean: batch_images, lr: learn_rate,
                                                       img_spec: special_input, special_gt: daub_Images,
                                                       training: True})
                    step += 1

                DnCNN_saver.save(sess, overwriting_path + DnCNN_model_name + str(epoch + 1) + ".ckpt")
                print("+++++ epoch " + str(epoch + 1) + " is saved successfully +++++")


if __name__ == '__main__':
    train()
