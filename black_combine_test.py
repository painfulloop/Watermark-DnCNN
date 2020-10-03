# -*- coding: utf-8 -*-
import os, cv2
import numpy as np
import DnCNN_model
import DeepPrior_black_model
import tensorflow as tf
import pylab, Psnr
import matplotlib.pyplot as plt
from matplotlib import cm
import P_value
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

np.random.seed(0)

sigma = 25
learn_rate = 0.0001
epochs = 1
batch_size = 128
train_data = './data/img_clean_pats.npy'
comb_model_path = './combine_weight/'
compress_path = './compress_weight/40/'
bsd_finetune_path = './finetune_weight_BSD200/'
org_finetune_path = './finetune_weight_org/'
overwriting_path = './overwriting/'
test_img_dir = './test_img'
input_data = './input_data/'


image_mod = 0
type = 'logo'

DIP_model_name= 'Black_DIP_' + type + '_weight_2'
DIP_model_path = comb_model_path
DnCNN_model_name= 'Black_DnCNN_' + 'logo' + '_weight_7'
# DnCNN_model_name= 'Black_DnCNN_texture_' + type + '_weight_fine_50'
# DnCNN_model_name= 'Black_DnCNN_BSD400_' + type + '_weight_fine_50'
# DnCNN_model_name= 'model_weight_45'
model_path = comb_model_path

def degradation(input, gt):
    num = np.prod(gt.shape)
    difference = np.sqrt(np.sum(np.square(input - gt))) / num
    print ('element degrade: ', difference)
    return difference


def his2D(input, gt):
    v1 = np.reshape(gt, np.prod(gt.shape))
    (n, bins) = np.histogram(v1, bins=50, normed=True)  # NumPy version (no plot)
    pylab.plot(1.0 * (bins[1:]), n)

    v2 = np.reshape(input, np.prod(input.shape))
    (n, bins2) = np.histogram(v2, bins=50, normed=True)  # NumPy version (no plot)
    pylab.plot(1.0 * (bins2[1:]), n)
    pylab.show()

def his3D(input, gt):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X = np.arange(1, 41, 1)
    Y = np.arange(1, 41, 1)
    X, Y = np.meshgrid(X, Y)
    Z_gt = np.squeeze(gt)
    Z_input = np.squeeze(input)

    surf = ax.plot_surface(X, Y, Z_gt, cmap=cm.coolwarm)
    fig.colorbar(surf, shrink=0.5, aspect=5)

    surf_input = ax.plot_surface(X, Y, Z_input, cmap='rainbow')
    fig.colorbar(surf_input, shrink=0.5, aspect=5)
    plt.show()

def post_process(input):

    input = np.squeeze(input, axis=0)
    input = input * 255
    input = np.clip(input, 0, 255)
    input = input.astype(np.uint8)
    input = np.squeeze(input, axis=2)

    return input



def transition(w):
    return w


def eval():
    with tf.Graph().as_default():

        img_clean= tf.placeholder(tf.float32, [None, None, None, 1], name='clean_image')

        #DnCNN model
        img_noise = img_clean + 0 * tf.random_normal(shape=tf.shape(img_clean), stddev=25 / 255.0)
        Y, N = DnCNN_model.dncnn(img_noise, is_training=False)


        #extract weight
        dncnn_s_out = transition(N)

        #DeepPrior model
        ldr = DeepPrior_black_model.Encoder_decoder(dncnn_s_out, is_training=True)


        dncnn_var_list = [v for v in tf.all_variables() if v.name.startswith('block')]
        DnCNN_saver = tf.train.Saver(dncnn_var_list)

        dip_var_list = [v for v in tf.all_variables() if v.name.startswith('DIP')]
        DIP_saver = tf.train.Saver(dip_var_list)

        with tf.Session() as sess:


            DnCNN_saver.restore(sess, model_path + DnCNN_model_name + ".ckpt")
            DIP_saver.restore(sess, DIP_model_path + DIP_model_name + ".ckpt")

            ramd_Image = cv2.imread('./input_data/spec_input.png', 0)
            ramd_Image = ramd_Image.astype(np.float32) / 255.0
            ramd_Image = np.expand_dims(ramd_Image, 0)
            ramd_Image = np.expand_dims(ramd_Image, 3)

            # ramd_Images = np.load('./spec_input/spec_14065.npy')
            # ramd_Image = ramd_Images[0,:, :, :]
            # ramd_Image = np.expand_dims(ramd_Image, 0)

            mid, out = sess.run([dncnn_s_out, ldr], feed_dict={img_clean: ramd_Image})
            print(mid.shape)

            mark_out = post_process(out)
            #
            # cv2.imshow('watermark', mark_out)
            # cv2.waitKey(0)



if __name__ == '__main__':
    eval()