import os
import numpy as np
import tensorflow as tf
import Preprocess_dataset_for_dncnn

import utility


def dncnn(input, is_training):
    with tf.variable_scope('block1'):
        output = tf.layers.conv2d(input, 64, 3, padding='same', activation=tf.nn.relu, trainable=False)
    for layers in range(2, 16 + 1):
        with tf.variable_scope('block%d' % layers):
            output = tf.layers.conv2d(output, 64, 3, padding='same', name='conv%d' % layers, use_bias=False,
                                      trainable=False)
            output = tf.nn.relu(tf.layers.batch_normalization(output, training=False))
    with tf.variable_scope('block17'):
        output = tf.layers.conv2d(output, 1, 3, padding='same', trainable=True)
    return input - output, output


def lossing(Y, GT, batch_size):
    loss = (1.0 / batch_size) * tf.nn.l2_loss(Y - GT)
    return loss


def optimizer(loss, lr):
    optimizer = tf.train.AdamOptimizer(lr, name='AdamOptimizer')
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        # train_op = optimizer.minimize(loss)
        var_list = [t for t in tf.trainable_variables()]
        var_list = var_list[-2:]  # add for freezing
        gradient = optimizer.compute_gradients(loss, var_list=var_list)
        train_op = optimizer.apply_gradients(gradient)
    return train_op


def transition(w):
    return w


def train(train_data='./data/img_clean_pats.npy', DnCNN_model_name='fineTuned_', epochs=8,
          batch_size=128, learn_rate=0.0001, sigma=25, org_model_path='./overwriting/',
          fineTuning_path='./fineTuning_weight/'):
    utility.create_folder(fineTuning_path)
    spec_size = [1, 40, 40, 1]
    save_ckpt_each = 5
    with tf.Graph().as_default():
        lr = tf.placeholder(tf.float32, shape=[], name='learning_rate')
        training = tf.placeholder(tf.bool, name='is_training')
        img_clean = tf.placeholder(tf.float32, [batch_size, spec_size[1], spec_size[2], spec_size[3]],
                                   name='clean_image')

        # DnCNN model
        img_noise = img_clean + tf.random_normal(shape=tf.shape(img_clean),
                                                 stddev=sigma / 255.0)  # dati con aggiunta di rumore

        Y, N = dncnn(img_noise, is_training=training)

        # host loss
        dncnn_loss = lossing(Y, img_clean, batch_size)

        dncnn_opt = optimizer(dncnn_loss, lr)
        init = tf.global_variables_initializer()

        dncnn_var_list = [v for v in tf.global_variables() if v.name.startswith('block')]
        DnCNN_saver = tf.train.Saver(dncnn_var_list, max_to_keep=50)

        np.random.seed(0)
        with tf.Session() as sess:
            data_total = np.load(train_data)
            data_total = data_total.astype(np.float32) / 255.0
            num_example, row, col, chanel = data_total.shape
            numBatch = num_example // batch_size
            sess.run(init)

            ckpt = tf.train.get_checkpoint_state(org_model_path)
            if ckpt and ckpt.model_checkpoint_path:
                full_path = tf.train.latest_checkpoint(org_model_path)
                print('last ckp', full_path)
                DnCNN_saver.restore(sess, full_path)
                print("Loading " + os.path.basename(full_path) + " to the model")
            else:
                print("DnCNN weight must be exist")
                assert ckpt != None, 'weights not exist'

            step = 0
            for epoch in range(1, epochs + 1):
                np.random.shuffle(data_total)
                for batch_id in range(0, numBatch):
                    batch_images = data_total[batch_id * batch_size:(batch_id + 1) * batch_size, :, :, :]
                    if batch_id % 100 == 0:
                        dncnn_loss_res = sess.run(dncnn_loss,
                                                  feed_dict={img_clean: batch_images, lr: learn_rate, training: False})
                    _ = sess.run(dncnn_opt, feed_dict={img_clean: batch_images, lr: learn_rate, training: True})
                    step += 1
                print(f"epoch={epoch}, step={step}, dncnn_loss={dncnn_loss_res}")
                if epoch % save_ckpt_each == 0:
                    DnCNN_saver.save(sess,
                                     os.path.join(fineTuning_path, DnCNN_model_name + str(epoch).zfill(2) + ".ckpt"))
                    print(f"++++ epoch {epoch} saved ++++")


if __name__ == '__main__':
    dataset = './data/img_clean_pats.npy'  # Fine tune with original datas
    # dataset = './data/img_clean_KTH_TIPS.npy' # Fine tune with different datas

    if dataset == './data/img_clean_pats.npy':
        train(train_data='./data/img_clean_pats.npy', epochs=100,
              fineTuning_path="fineTuning_weights_Img12")  # Fine tune with original datas
    elif dataset == './data/img_clean_KTH_TIPS.npy':
        if not (os.path.isfile(dataset)):
            Preprocess_dataset_for_dncnn.generate_patches(bat_size=100, step=40, stride=80, pat_size=40,
                                                          out_name='img_clean_KTH_TIPS', save_dir='./data',
                                                          src_dir='./dataset/Train_KTH_TIPS')

        train(train_data='./data/img_clean_KTH_TIPS.npy', epochs=100,
              fineTuning_path="fineTuning_weights_KTH")
