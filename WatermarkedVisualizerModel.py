# -*- coding: utf-8 -*-
import cv2
import numpy as np
import DnCNNModel
import AuxVisualizerModel
import tensorflow as tf
import pylab
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import utility
import os

np.random.seed(0)


def degradation(input, gt):
    num = np.prod(gt.shape)
    difference = np.sqrt(np.sum(np.square(input - gt))) / num
    print('element degrade: ', difference)
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


class WatermarkedVisualizerModel(object):
    def __init__(self):
        self.loaded = False
        self.session: tf.Session = None
        self.img_clean = None
        self.img_noise = None
        self.Y, self.N = None, None
        self.training_placeholder = None
        self.ldr = None
        self.dncnn_s_out = None

    def build_model(self, model_path='./DnCNN_weight/', DnCNN_model_name='model_weight_45'):
        DIP_model_path = './combine_weight/'
        DIP_model_name = 'Black_DIP_sign_weight_8'
        if self.loaded:
            self.session.close()
            del self.session
            del self.training_placeholder
            del self.img_noise, self.img_clean
            del self.Y, self.N
            del self.ldr, self.dncnn_s_out
        # with tf.Graph().as_default():
        self.img_clean = tf.placeholder(tf.float32, [None, None, None, 1], name='clean_image')
        self.training_placeholder = tf.placeholder(tf.bool, name='is_training')

        # DnCNN model
        img_noise = self.img_clean + 0 * tf.random_normal(shape=tf.shape(self.img_clean),
                                                          stddev=25 / 255.0)  # trigger img
        self.Y, self.N = DnCNNModel.dncnn(img_noise, is_training=False)

        # extract weight
        self.dncnn_s_out = transition(self.N)

        # DeepPrior model
        self.ldr = AuxVisualizerModel.Encoder_decoder(self.dncnn_s_out, is_training=True)

        dncnn_var_list = [v for v in tf.all_variables() if v.name.startswith('block')]
        DnCNN_saver = tf.train.Saver(dncnn_var_list)

        dip_var_list = [v for v in tf.all_variables() if v.name.startswith('DIP')]
        DIP_saver = tf.train.Saver(dip_var_list)

        self.session = tf.Session()
        DnCNN_saver.restore(self.session, os.path.join(model_path, DnCNN_model_name + ".ckpt"))
        DIP_saver.restore(self.session, os.path.join(DIP_model_path, DIP_model_name + ".ckpt"))
        self.loaded = True

    def eval(self, trigger_image='trigger_image.png', show_input=True):
        if not self.loaded:
            print("Model not loaded. load it to start")
            return np.zeros((40, 40))
        if type(trigger_image) is str:
            ramd_Image = cv2.imread('key_imgs/' + trigger_image, 0)
        else:
            ramd_Image = trigger_image

        ramd_Image = ramd_Image.astype(np.float32) / 255.0
        ramd_Image = np.expand_dims(ramd_Image, 0)
        ramd_Image = np.expand_dims(ramd_Image, 3)

        # ramd_Images = np.load('./spec_input/spec_14065.npy')
        # ramd_Image = ramd_Images[0,:, :, :]
        # ramd_Image = np.expand_dims(ramd_Image, 0)

        mid, out = self.session.run([self.dncnn_s_out, self.ldr],
                                    feed_dict={self.img_clean: ramd_Image, self.training_placeholder: False})
        # print(mid.shape)

        mark_out = post_process(out)
        return mark_out


if __name__ == '__main__':
    out_copyrightImg_path = 'out_copyrightImg'
    utility.create_folder(out_copyrightImg_path)

    # comment here to change source model.'DnCNN_weight' is original model, 'overwrting' is WM trained model
    # model_path = './DnCNN_weight/'
    model_path = './overwriting/'
    dip_model_path = './combine_weight/'

    model = WatermarkedVisualizerModel()
    model.build_model(model_path=model_path, DnCNN_model_name=utility.get_last_model(model_path))

    img = model.eval()
    cv2.imwrite(out_copyrightImg_path + '/copyrightImg.png', img)
    utility.show_image(img, title='watermark')
