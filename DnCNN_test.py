# -*- coding: utf-8 -*-
import numpy as np
import DnCNNModel
import cv2
import tensorflow as tf


def ssim(img1, img2):
    k1 = 0.01
    k2 = 0.03
    L = 255
    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2

    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    img1_2 = img1 * img1
    img2_2 = img2 * img2
    img1_img2 = img1 * img2

    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)

    mu1_2 = mu1 * mu1
    mu2_2 = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_2 = cv2.GaussianBlur(img1_2, (11, 11), 1.5)
    sigma1_2 -= mu1_2

    sigma2_2 = cv2.GaussianBlur(img2_2, (11, 11), 1.5)
    sigma2_2 -= mu2_2

    sigma12 = cv2.GaussianBlur(img1_img2, (11, 11), 1.5)
    sigma12 -= mu1_mu2

    t1 = 2 * mu1_mu2 + c1
    t2 = 2 * sigma12 + c2
    t3 = t1 * t2

    t1 = mu1_2 + mu2_2 + c1
    t2 = sigma1_2 + sigma2_2 + c2
    t1 = t1 * t2

    ssim = t3 / t1
    mean_ssim = np.mean(ssim)

    return mean_ssim

def psnr(img1, img2):
    img1 = np.clip(img1, 0, 255)

    img2 = np.clip(img2, 0, 255)

    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    if (len(img1.shape) == 2):
        m, n = img1.shape
        k = 1
    elif (len(img1.shape) == 3):
        m, n, k = img1.shape

    B = 8
    diff = np.power(img1 - img2, 2)
    MAX = 2 ** B - 1
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


def eval(model_name='model_weight_45', model_path='./DnCNN_weight/', test_img='./dataset/test/Set12/01.png', sigma=25, show_imput=True):
    with tf.Graph().as_default():
        img_clean = tf.placeholder(tf.float32, [None, None, None, 1], name='clean_image')
        training = tf.placeholder(tf.bool, name='is_training')
        img_noise = img_clean + tf.random_normal(shape=tf.shape(img_clean), stddev=sigma / 255.0)
        Y, N = DnCNNModel.dncnn(img_noise, is_training=training)

        dncnn_var_list = [v for v in tf.all_variables() if v.name.startswith('block')]
        DnCNN_saver = tf.train.Saver(dncnn_var_list)

        with tf.Session() as sess:
            DnCNN_saver.restore(sess, model_path + model_name + ".ckpt")
            if type(test_img) is str:
                img_raw = cv2.imread(test_img, 0)
            else:
                img_raw = test_img
            # img_raw = np.random.randint(0, 255, size=(256,256))
            img = img_raw.astype(np.float) / 255
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
            # print(model_name + ' psnr: ' + str(psnr(out, img_raw)))
            if show_imput:
                cv2.imshow('outDenoiseImg', out)
                cv2.imshow('noise', n)
                cv2.imshow('img_noising', img_n)
                cv2.waitKey(0)
            return out


if __name__ == '__main__':
    eval()
