# -*- coding: utf-8 -*-
import numpy as np
import DnCNN_model
import cv2
import tensorflow as tf

sigma = 25
model_id = 7
model_path = './combine_weight/'
test_img = './dataset/dncnn-img/test/Set12/01.png'
# test_img = './input_data/spec_input.png'
compress_path = './compress_weight/25/'
model_name = 'Black_DnCNN_logo_weight_' + str(model_id)
# model_name = 'model_weight_45'

def psnr(img1, img2):
    img1 = np.clip(img1, 0, 255)

    img2 = np.clip(img2, 0, 255)

    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    if(len(img1.shape) == 2):
        m, n = img1.shape
        k = 1
    elif (len(img1.shape) == 3):
        m, n, k = img1.shape

    B = 8
    diff = np.power(img1 - img2, 2)
    MAX = 2**B - 1
    MSE = np.sum(diff) / (m * n * k)
    sqrt_MSE = np.sqrt(MSE)
    PSNR = 20 * np.log10(MAX / sqrt_MSE)

    return PSNR

def post_process(img):
    img = np.squeeze(img)
    img = img * 255
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)
    return img

def eval():
    with tf.Graph().as_default():
        img_clean = tf.placeholder(tf.float32, [None, None, None, 1], name='clean_image')
        training = tf.placeholder(tf.bool, name='is_training')
        img_noise = img_clean + tf.random_normal(shape=tf.shape(img_clean), stddev=sigma / 255.0)
        Y, N = DnCNN_model.dncnn(img_noise, is_training=training)

        dncnn_var_list = [v for v in tf.all_variables() if v.name.startswith('block')]
        DnCNN_saver = tf.train.Saver(dncnn_var_list)

        with tf.Session() as sess:
            DnCNN_saver.restore(sess, model_path + model_name +".ckpt")

            img_raw = cv2.imread(test_img, 0)
            # img_raw = np.random.randint(0, 255, size=(256,256))
            img = img_raw.astype(np.float)/255
            img = np.expand_dims(img, axis=0)
            img = np.expand_dims(img, axis=3)

            # img =25 * np.random.normal(size=[1, 256, 256, 1]) / 255
            # img_raw = img

            out, n, img_n = sess.run([Y, N, img_noise], feed_dict={img_clean: img, training: False})
            out = post_process(out)
            n = post_process(n)
            img_n = post_process(img_n)

            # different = np.sum(np.abs(img_n - n))
            # print (different)
            print ('psnr: ', psnr(out, img_raw))

            cv2.imshow('out', out)
            cv2.imshow('n', n)
            cv2.imwrite('out_pa.png', out)
            cv2.imshow('img_n', img_n)
            cv2.waitKey(0)

if __name__ == '__main__':
    eval()



