import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import os

# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.python.client import device_lib
tf.logging.set_verbosity(tf.logging.WARN)
import logging
logging.getLogger('tensorflow').disabled = True

def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]



print("\n    ---- DEVICES USED BY TENSORFLOW ---- \n")
print(get_available_devices())
# preprocess(directory)
filepaths_new = glob.glob("/home/s2936860/CAT-GAN/mega-cat-dataset")
path_samples = "/data/s2936860/DCGAN-var3-1"
path_final = "/data/s2936860/DCGAN-final-var3-1"


print("\n    ---- CREATING FOLDER FOR OUTPUT IMAGES ---- ")

if not os.path.exists(path_samples):
    try:
        os.mkdir(path_samples)
    except:
        raise OSError("Can't create destination directory (%s)!" % (path_samples))  


if not os.path.exists(path_final):
    try:
        os.mkdir(path_final)
    except:
        raise OSError("Can't create destination directory (%s)!" % (path_final))  


print("\n    ---- LOADING THE DATA ---- ")

imagenames = []
for folder in filepaths_new:
    for f in glob.glob(folder + '/*'):
        imagenames.append(f)

i = 0
all_dat = []
for image in imagenames:
    # open image
    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = np.resize(img, (64, 64, 3))
    # img = np.expand_dims(img, 0)
    # print(img.shape)
    all_dat.append(img)
    # print(i)
    i = i + 1

# shows images details
print("\nImages found: ", len(all_dat))
print("Size images: ", all_dat[0].shape)



MOMENTUM_K = 0.99


def next_batch(num, data=all_dat):
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    shuffled = np.asarray(data_shuffle)
    return np.asarray(shuffled)

keep_prob_train = 0.5  # 0.5
batch_size = 64
n_noise = 64 #size noise vector

# generate noise for making the epic GIF
gifnoise = np.random.uniform(0.0, 1.0, [batch_size, n_noise]).astype(np.float32)


def montage(images):
    if isinstance(images, list):
        images = np.array(images)
        img_h = images.shape[1]
        img_w = images.shape[2]
        n_plots = int(np.ceil(np.sqrt(images.shape[0])))
        if len(images.shape) == 4 and images.shape[3] == 3:
            m = np.ones(
                (images.shape[1] * n_plots + n_plots + 1,
                 images.shape[2] * n_plots + n_plots + 1, 3)) * 0.5
        else:
            raise ValueError('Could not parse image shape of {}'.format(
                images.shape))
        for i in range(n_plots):
            for j in range(n_plots):
                this_filter = i * n_plots + j
                if this_filter < images.shape[0]:
                    this_img = images[this_filter]
                    m[1 + i + i * img_h:1 + i + (i + 1) * img_h,
                    1 + j + j * img_w:1 + j + (j + 1) * img_w] = this_img
        return m


tf.reset_default_graph()

X_in = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 3], name='X')
noise = tf.placeholder(dtype=tf.float32, shape=[None, n_noise])

keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')  # for later dropout?
is_training = tf.placeholder(dtype=tf.bool, name='is_training')  # batch normalization?


def lrelu(x):
    return tf.maximum(x, tf.multiply(x, 0.2))


def binary_cross_entropy(x, z):  # binary as discriminator deals with binary prokeep_prob_trainblem
    eps = 1e-12
    return (-(x * tf.log(z + eps) + (1. - x) * tf.log(1. - z + eps)))

# Discriminator with increasing filter size
def discriminator(img_in, reuse=None, keep_prob=keep_prob):
    activation = lrelu
    with tf.variable_scope("discriminator", reuse=reuse):
        x = tf.reshape(img_in, shape=[-1, 64, 64, 3])
        x = tf.layers.conv2d(x, kernel_size=4, filters=256, strides=2, padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, kernel_size=4, filters=512, strides=2, padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, kernel_size=4, filters=1024, strides=2, padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob)
        x = tf.contrib.layers.flatten(x)
            # x = tf.layers.dense(x, units=128, activation=activation)
        x = tf.layers.dense(x, units=1, activation=tf.nn.sigmoid)
        return x



def generator(z, keep_prob=keep_prob, is_training=is_training):
    activation = lrelu
    momentum = MOMENTUM_K
    with tf.variable_scope("generator", reuse=None):
        x = z
        d1 = 4
        d2 = 3
        x = tf.layers.dense(x, units=d1 * d1 * d2, activation=activation)
        x = tf.layers.dropout(x, keep_prob)
        x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)
        x = tf.reshape(x, shape=[-1, d1, d1, d2])
        x = tf.image.resize_images(x, size=[16, 16])
        x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=512, strides=2, padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob)
        x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)
        x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=256, strides=2, padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob)
        x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)
        x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=128, strides=1, padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob)
        x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)
        x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=64, strides=1, padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob)
        x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)
        x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=3, strides=1, padding='same', activation=tf.nn.sigmoid)
        return x

g = generator(noise, keep_prob, is_training)  # create generator with input noise variable
print(g)

d_real = discriminator(X_in)  # make discriminator real object compatible with desired image vars
d_fake = discriminator(g,
                           reuse=True)  # make discriminator fake object (generator output) compatible with the same vars as real images

vars_g = [var for var in tf.trainable_variables() if
              var.name.startswith("generator")]  # obtain variables of a generator
vars_d = [var for var in tf.trainable_variables() if
              var.name.startswith("discriminator")]  # obtain variables of a discriminator

d_reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-6),
                                                   vars_d)  # apply weight-decay to discriminator
g_reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-6),
                                                   vars_g)  # apply weight-decay to generator

loss_d_real = binary_cross_entropy(tf.ones_like(d_real), d_real)  # loss of a real input to the discriminator should be predicted as 1 in BCE
loss_d_fake = binary_cross_entropy(tf.zeros_like(d_fake), d_fake)  # loss of a generated input to the discriminator should be predicted as 0 in BCE
loss_g = tf.reduce_mean(binary_cross_entropy(tf.ones_like(d_fake), d_fake))  # loss of the generator is to try to tend the BCE of discriminator fake images input to 1 so actually 1 - loss_d_fake

loss_d = tf.reduce_mean(0.5 * (loss_d_real + loss_d_fake))  # loss discriminator is combination of fake and real loss /2

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # use update_ops as we are using batch normalization

with tf.control_dependencies(update_ops):
    optimizer_d = tf.train.RMSPropOptimizer(learning_rate=0.00005).minimize(loss_d + d_reg, var_list=vars_d)
        
    optimizer_g = tf.train.RMSPropOptimizer(learning_rate=0.0002).minimize(loss_g + g_reg,  var_list=vars_g) 
print("\n    ---- INITIALIZING VARIABLES ---- ")

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print("\n    ---- LEARNING IN PROCESS ---- ")
for i in range(15000):


    n = np.random.uniform(0.0, 1.0, [batch_size, n_noise]).astype(np.float32)
    batch = [b for b in next_batch(num=batch_size)]

    d_real_ls, d_fake_ls, g_ls, d_ls = sess.run([loss_d_real, loss_d_fake, loss_g, loss_d],
                                                    feed_dict={X_in: batch,
                                                               noise: n,
                                                               keep_prob: keep_prob_train,
                                                               is_training: True})
    d_fake_ls_init = d_fake_ls
    d_real_ls = np.mean(d_real_ls)
    d_fake_ls = np.mean(d_fake_ls)
    g_ls = g_ls
    d_ls = d_ls


    sess.run(optimizer_d, feed_dict={noise: n, X_in: batch, keep_prob: keep_prob_train, is_training: True})

    sess.run(optimizer_g, feed_dict={noise: n, keep_prob: keep_prob_train, is_training: True})




    # generate random cats
    if not i % 4999:
        for z in range(64):
            n = np.random.uniform(0.0, 1.0, [batch_size, n_noise]).astype(np.float32)
            gen_img = sess.run(g, feed_dict={noise: n, keep_prob: 1.0, is_training: False})
            imgs = [img[:, :, :] for img in gen_img]
            #gen_img = montage(imgs)
            ctr = 0
            for img in imgs:
                plt.imsave(path_samples + "/cat_epoch" +str(i)+'_n'+str(z)+ '_n'+ str(ctr)+ '.png', img)
                ctr += 1


    # generate random cats
for z in range(40000):
    n = np.random.uniform(0.0, 1.0, [batch_size, n_noise]).astype(np.float32)
    gen_img = sess.run(g, feed_dict={noise: n, keep_prob: 1.0, is_training: False})
    imgs = [img[:, :, :] for img in gen_img]
    #gen_img = montage(imgs)
    ctr = 0
    for img in imgs:
            plt.imsave(path_final + "/cat_finals" +str(z)+ '_n'+ str(ctr) + '.png', img)
            ctr += 1
