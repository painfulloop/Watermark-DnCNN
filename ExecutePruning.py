import tensorflow as tf
from tensorflow import keras
import os, cv2
import numpy as np
import DnCNNModel
from prunings import unit_pruning, weight_pruning, pruning_factory

np.random.seed(0)
lambda_ = 0.001
spec_size = [1, 40, 40, 1]

from tensorflow import keras


def dncnn_keras(input)->keras.Sequential:
    model=keras.Sequential()
    model.add(keras.layers.Conv2D(input, 64, 3, padding='same', activation=tf.nn.relu))
    for layers in range(2, 16 + 1):
        model.add(keras.layers.Conv2D(input, 64, 3, padding='same', activation=tf.nn.relu))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation(keras.activations.relu))
    model.add(keras.layers.Conv2D(1, 3, padding='same'))
    return model


def load_and_prune_model(org_model_path='./DnCNN_weight/', model_name="Black_DnCNN_cman_weight_8", out_pruned_path='./pruning_weights',k=.1, batch_size=128, learn_rate=0.0001, sigma=25):

    # './DnCNN_weight/' folder containing weights of original DnCNN
    # './overwriting/' folder containing new weights created in this script ( model trained with trigger key).
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
        img_noise = img_clean + tf.random_normal(shape=tf.shape(img_clean),
                                                 stddev=sigma / 255.0)  # dati con aggiunta di rumore
        img_total = tf.concat([img_noise, img_spec], 0)  # concatenazione img_noise e img trigger
        Y, N = DnCNNModel.dncnn(img_total, is_training=training)

        # slide
        Y_img = tf.slice(Y, [0, 0, 0, 0], [batch_size, spec_size[1], spec_size[2], spec_size[3]])
        N_spe = tf.slice(N, [batch_size, 0, 0, 0], [special_num, spec_size[1], spec_size[2], spec_size[3]])

        # host loss
        dncnn_loss = DnCNNModel.lossing(Y_img, img_clean, batch_size)

        dncnn_var_list = [v for v in tf.global_variables() if v.name.startswith('block')]
        DnCNN_saver = tf.train.Saver(dncnn_var_list, max_to_keep=50)

        # Real pruning is done here before session init and run
        updates = []
        for layer in dncnn_var_list:#reversed(dncnn_var_list):#[-5:]:
            if "conv" in layer.name:
                update_op = weight_pruning(layer, k)
                updates.append(update_op)

        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)

            ckpt = tf.train.get_checkpoint_state(org_model_path)

            if ckpt and ckpt.model_checkpoint_path:
                full_path = os.path.join(org_model_path, model_name)
                print(full_path)
                DnCNN_saver.restore(sess, full_path)
                print("Loading " + os.path.basename(full_path) + " to the model")

            else:
                print("DnCNN weight must be exist")
                assert ckpt != None, 'weights not exist'

            sess.run(updates)
            out_path = os.path.join(out_pruned_path, "Pruned_k{:.2f}.ckpt".format(k))
            DnCNN_saver.save(sess, out_path)
            print("Pruned model succesfully saved in " + out_path)


if __name__ == '__main__':
    import utility
    utility.create_folder("pruning_weights")
    for value_k in np.arange(0.05, 0.61, 0.05):
        k = round(float(value_k), 2)
        load_and_prune_model(org_model_path="overwriting", model_name=utility.get_last_model("overwriting")+".ckpt", out_pruned_path="pruning_weights", k=k)
        print("Pruned {}".format(k))
    #prune_model(org_model_path=os.path.join("overwriting", utility.get_last_model("overwriting")), out_pruned_path="pruning_weights/pruned_float16")