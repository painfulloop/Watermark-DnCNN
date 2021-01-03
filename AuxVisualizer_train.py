# -*- coding: utf-8 -*-
import os, cv2
import numpy as np
import DnCNNModel
import AuxVisualizerModel
import tensorflow as tf

np.random.seed(0)

# comment here to change source model.'DnCNN_weight' is original model, 'overwrting' is WM trained model
# org_model_path = './DnCNN_weight/'

image_mod = 0
type = 'sign'
daub_size = [320, 320, 2 * image_mod + 1]

DIP_model_name = 'Black_DIP_' + type + '_weight_'


def post_process(input, step):
    # print (input.shape)
    input = np.squeeze(input, axis=0)
    input = input * 255
    input = np.clip(input, 0, 255)
    input = input.astype(np.uint8)
    input = np.squeeze(input, axis=2)

    cv2.imwrite('./temp/gt_' + str(step) + '.png', input)


def transition(w):
    return w


def ft_DIP_optimizer(loss, lr):
    optimizer = tf.train.AdamOptimizer(0.001, name='AdamOptimizer_DIP')
    ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    tensors = [k for k in ops if k.name.startswith('DIP')]
    with tf.control_dependencies(tensors):
        var_list = [t for t in tf.all_variables() if t.name.startswith('DIP')]
        gradient = optimizer.compute_gradients(loss, var_list=var_list)
        train_op = optimizer.apply_gradients(gradient)
    return train_op


def train(train_data='./data/img_clean_pats.npy', org_model_path='./overwriting/', comb_model_path='./combine_weight/',
          test_img_dir='./test_img',
          epochs=8, batch_size=128, learn_rate=0.001, sigma=25):
    degraded_image = os.path.join(test_img_dir, type + '.png')  # copyright img
    special_num = 20
    with tf.Graph().as_default():
        lr = tf.placeholder(tf.float32, shape=[], name='learning_rate')
        tag = tf.placeholder(tf.float32, shape=[], name='tag')
        training = tf.placeholder(tf.bool, name='is_training')
        img_clean = tf.placeholder(tf.float32, [None, None, None, 1], name='clean_image')

        images_daub = tf.placeholder(tf.float32, [None, daub_size[0], daub_size[1], daub_size[2]])

        # #DnCNN model
        img_noise = img_clean + tag * tf.random_normal(shape=tf.shape(img_clean),
                                                       stddev=sigma / 255.0)  # img clean = trigger img
        Y, N = DnCNNModel.dncnn(img_noise, is_training=training)
        dncnn_loss = DnCNNModel.lossing(Y, img_clean, batch_size)

        # extract weight
        dncnn_s_out = transition(N)

        # DeepPrior model
        ldr = AuxVisualizerModel.Encoder_decoder(dncnn_s_out, is_training=True)  # dncnn_s_out = verification img
        dip_loss = AuxVisualizerModel.lossing(ldr, images_daub)

        # Update DIP model
        dip_opt = ft_DIP_optimizer(dip_loss, lr)

        init = tf.global_variables_initializer()

        dncnn_var_list = [v for v in tf.global_variables() if v.name.startswith('block')]
        DnCNN_saver = tf.train.Saver(dncnn_var_list)

        dip_var_list = [v for v in tf.all_variables() if v.name.startswith('DIP')]
        DIP_saver = tf.train.Saver(dip_var_list, max_to_keep=50)

        with tf.Session() as sess:
            data_total = np.load(train_data)
            data_total = data_total.astype(np.float32) / 255.0
            num_example, row, col, chanel = data_total.shape
            numBatch = num_example // batch_size

            daub_Images = cv2.imread(degraded_image, 0)  # copyright img
            daub_Images = cv2.resize(daub_Images, (daub_size[0], daub_size[1]))
            daub_Images = daub_Images.astype(np.float32) / 255
            daub_Images = np.expand_dims(daub_Images, axis=0)
            daub_Images = np.expand_dims(daub_Images, axis=3)
            # daub_Images = np.repeat(daub_Images, special_num, axis=0)

            # special_input = cv2.imread('./input_data/spec_input.png', 0)  # trigger img
            special_input = cv2.imread('key_imgs/trigger_image.png', 0)
            special_input = special_input.astype(np.float32) / 255.0
            special_input = np.expand_dims(special_input, 0)
            special_input = np.expand_dims(special_input, 3)
            # special_input = np.repeat(special_input, special_num, axis=0)

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

                    __ = sess.run(dip_opt, feed_dict={img_clean: special_input, lr: learn_rate,
                                                      images_daub: daub_Images, tag: 0.0,
                                                      training: False})
                    step += 1

                    if batch_id % 100 == 0:
                        dip_lost = sess.run(dip_loss, feed_dict={img_clean: special_input, lr: learn_rate,
                                                                 images_daub: daub_Images, tag: 0.0,
                                                                 training: False})

                        print("step = %d, dncnn_loss = %f, dip_loss = %f" % (step, 0, dip_lost))

                save_path = DIP_saver.save(sess, comb_model_path + DIP_model_name + str(epoch + 1) + ".ckpt")
                print("+++++ epoch " + str(epoch + 1) + " is saved successfully +++++")


if __name__ == '__main__':
    train()
