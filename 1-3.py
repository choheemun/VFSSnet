
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
# libs/datasets/dataset_factory.py
# from libs.visualization.summary_utils import visualize_input
import glob
import time
from tensorflow.python.lib.io.tf_record import TFRecordCompressionType
from tensorflow.python import keras
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras import backend as k
from tensorflow.keras.layers import Dense
from tensorflow.keras.metrics import categorical_accuracy as accuracy
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Input, Activation, Conv2D, Conv2DTranspose, Flatten, Dropout, MaxPool2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.losses import categorical_crossentropy  # tensorflow.keras.objectives는 tf 1.12에는 없는듯


a=np.array([[1,2]])

#image=tf.cast(a,tf.float32)




graph_1 = tf.Graph()

with graph_1.as_default():

    x = tf.placeholder(tf.float32, shape=(None, 2))

    m = Dense(20, activation='relu', kernel_initializer='he_normal')(x)
    m=m/10
    m = Dense(100, activation='relu', kernel_initializer='he_normal')(m)
    m=m/10
    m = Dense(100, activation='relu', kernel_initializer='he_normal')(m)
    m=m/10
    m = Dense(100, activation='relu', kernel_initializer='he_normal')(m)
    m=m/10
    pred = Dense(4, activation='sigmoid', kernel_initializer='he_normal')(m)
    pred=pred*4






    #loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(labels, pred))
    #train_step = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

    sess=tf.Session()


    global_step = tf.Variable(0, trainable=False, name='global_step')

    init_op = tf.global_variables_initializer()
    sess.run(init_op)




with tf.Session(graph=graph_1) as sess:
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=20)
    ckpt = tf.train.get_checkpoint_state('dataset/checkpoint/checkpoint11')

    saver.restore(sess, "dataset/checkpoint/checkpoint11/nn.ckpt-0")
    pred_val=sess.run([pred], feed_dict={x: a})
    print(pred_val)
    #checkpoint_path = os.path.join('dataset/checkpoint/checkpoint11', 'nn.ckpt')
    #saver.save(sess, checkpoint_path, global_step=global_step)

    sess.close()




pred_val=np.array(pred_val)
pred_val = np.squeeze(pred_val, 0)

print(pred_val.shape)



















#image=tf.cast(a,tf.float32)
graph_2 = tf.Graph()

with graph_2.as_default():


    x1 = tf.placeholder(tf.float32, shape=(None, 4))

    m = Dense(20, activation='relu', kernel_initializer='he_normal')(x1)
    m=m/10
    m = Dense(200, activation='relu', kernel_initializer='he_normal')(m)
    m=m/10
    m = Dense(200, activation='relu', kernel_initializer='he_normal')(m)
    m=m/10
    m = Dense(200, activation='relu', kernel_initializer='he_normal')(m)
    m=m/10
    pred = Dense(6, activation='sigmoid', kernel_initializer='he_normal')(m)
    pred=pred*6


    sess=tf.Session()


    global_step2 = tf.Variable(0, trainable=False, name='global_step')

    init_op = tf.global_variables_initializer()
    sess.run(init_op)


with tf.Session(graph=graph_2) as sess:
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=20)
    ckpt2 = tf.train.get_checkpoint_state('dataset/checkpoint/checkpoint12')

    saver.restore(sess, "dataset/checkpoint/checkpoint12/nn.ckpt-0")

    pred_val=sess.run([pred], feed_dict={x1: pred_val})
    print(pred_val)


    sess.close()
