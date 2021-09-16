'''
test data 이외에 inference data가 따로 있어야 하며
inference data를 이미 tfrecord파일로 변환해두어야 한다   --------> 그냥 feeddict 로 넣자
각 NN
즉 train_mask, train_pharyngeal, train_hyoid 모델을
모두 inference.py에 두고
각각의 check point 파일들을 불러 inference 시켜 결과를 얻으면 될듯

augmentation 제거
pipeline제거  대신 feeddict 넣자
random 요소 제거


grey scale이므로 newaxis만들어준다. (tf.expand_dim)

preprocess한다.

sess.run(box)로 값을 얻어내고 리스트에 추가한다
(optimize 자체를 안한다.)


이후 시각화~~~~


'''
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
from PIL import Image
import cv2
from math import atan2, degrees
import math
import matplotlib.pyplot as plt


# libs/datasets/dataset_factory.py


def preprocess_image(image, is_training=False):
    """preprocess image for coco
    1. random flipping
    2. min size resizing
    3. zero mean          ----- 이것만
    4. ...
    """

    #image = tf.cast(image, tf.float32)
    image = image / 256.0
    image = (image - 0.5) * 2.0
    # image = tf.expand_dims(image, axis=0)     # batch 때문에 늘리는듯

    return image


def root_mean_squared_error(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.losses.mean_squared_error(y_true, y_pred)))





def angle_between(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    deg1 = (360 + degrees(atan2(x1 - x2, y1 - y2))) % 360
    deg2 = (360 + degrees(atan2(x3 - x2, y3 - y2))) % 360
    return deg2 - deg1 if deg1 <= deg2 else 360 - (deg1 - deg2)






def rotation(x,y,degree):
    rad = degree * (math.pi / 180.0)
    nx = (math.cos(rad) * x - math.sin(rad) * y)
    ny = (math.sin(rad) * x + math.cos(rad) * y)
    return nx, ny






def inference():
    graph_points1 = []
    graph_points2 = []







    x = tf.placeholder(tf.float32, shape=[1, 448, 448, 3])

    model = ResNet50(input_tensor=x, include_top=False, weights='imagenet', pooling='max')
    model.trainable = False

    selected_model1 = Model(inputs=model.input,
                            outputs=model.get_layer('activation_48').output)  # output=(None, 14, 14, 2048)
    selected_model2 = Model(inputs=model.input,
                            outputs=model.get_layer('activation_39').output)  # output=(None, 28, 28, 1024)
    selected_model3 = Model(inputs=model.input,
                            outputs=model.get_layer('activation_21').output)  # output=(None, 56, 56, 512)
    selected_model4 = Model(inputs=model.input,
                            outputs=model.get_layer('activation_9').output)  # output=(None, 112, 112, 256)
    # selected_model1 ~ 3 으로만 pyramid를 build 하자

    y1 = selected_model1.output
    y2 = selected_model2.output
    y3 = selected_model3.output
    y4 = selected_model4.output

    pyramid_1 = Conv2D(filters=256, strides=(1, 1), kernel_size=(1, 1), activation='relu',
                       kernel_initializer='he_normal', padding='same', name="P1")(y1)  # (None, 14, 14, 256)

    pre_P2_1 = tf.image.resize_bilinear(pyramid_1, [28, 28], name='pre_P2_1')  # (None, 28, 28, 256)
    pre_P2_2 = Conv2D(filters=256, strides=(1, 1), kernel_size=(1, 1), activation='relu',
                      kernel_initializer='he_normal', padding='same', name="pre_p2_2")(y2)  # (None, 28, 28, 256)
    pre_P2_2 = BatchNormalization()(pre_P2_2)
    pre_P2 = tf.add(pre_P2_1, pre_P2_2)  # (None, 28, 28, 256)

    pyramid_2 = Conv2D(filters=256, strides=(1, 1), kernel_size=(3, 3), activation='relu',
                       kernel_initializer='he_normal', padding='same', name="P2")(pre_P2)  # (None, 28, 28, 256)
    pyramid_2 = BatchNormalization()(pyramid_2)

    pre_P3_1 = tf.image.resize_bilinear(pyramid_2, [56, 56], name='pre_P3_1')  # (None, 56, 56, 256)
    pre_P3_2 = Conv2D(filters=256, strides=(1, 1), kernel_size=(1, 1), activation='relu',
                      kernel_initializer='he_normal', padding='same', name="pre_p3_2")(y3)  # (None, 56, 56, 256)
    pre_P3_2 = BatchNormalization()(pre_P3_2)

    pre_P3 = tf.add(pre_P3_1, pre_P3_2)  # (None, 56, 56, 256)
    pyramid_3 = Conv2D(filters=256, strides=(1, 1), kernel_size=(3, 3), activation='relu',
                       kernel_initializer='he_normal', padding='same', name="P3")(pre_P3)  # (None, 56, 56, 256)
    pyramid_3 = BatchNormalization()(pyramid_3)
    m = pyramid_3

    # pre_P4_1= =tf.image.resize_bilinear(pyramid_2, [112,112], name='pre_P4_1')                                                   #(None, 112, 112, 256)
    # pre_P4_2 = Conv2D(filters=256, strides=(1, 1), kernel_size=(1, 1), activation='relu', padding='same', name="pre_p4_2")(y4)   #(None, 112, 112, 256)
    # pre_P4=tf.add(pre_P4_1,pre_P4_2)                                                                                             #(None, 112, 112, 256)
    # pyramid_4=Conv2D(filters=256, strides=(1, 1), kernel_size=(3, 3), activation='relu', padding='same', name="P4")(pre_P4)      #(None, 112, 112, 256)

    for _ in range(3):  # (None, 56, 56, 256)-> (None, 7,7,512)
        for _ in range(3):
            m = Conv2D(filters=512, strides=(1, 1), kernel_size=(3, 3), activation='relu',
                       kernel_initializer='he_normal', padding='same', name="C1")(m)
            m = BatchNormalization()(m)

        m = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(m)

    m = Flatten()(m)  # (None,7,7,512) -> (none, 39424)
    m = Dense(1024, activation='relu', kernel_initializer='he_normal')(m)
    m = BatchNormalization()(m)
    m = m / 100
    m = Dropout(0.3)(m)
    m = Dense(1024, activation='relu', kernel_initializer='he_normal')(m)
    m = BatchNormalization()(m)
    m = m / 100
    m = Dropout(0.3)(m)
    m = Dense(256, activation='relu', kernel_initializer='he_normal')(m)
    m = BatchNormalization()(m)
    m = m / 100
    box = Dense(4, activation='sigmoid', kernel_initializer='glorot_normal')(m)

    global_step = tf.Variable(0, trainable=False, name='global_step')
    # loss = root_mean_squared_error(gt_box, box)  # (None, 4)  vs  (None, 4)
    # train_step = tf.train.AdamOptimizer(learning_rate=0.000001).minimize(loss, global_step=global_step)

    sess = tf.Session()
    k.set_session(sess)

    with tf.Session() as sess:
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=20)
        # ckpt = tf.train.get_checkpoint_state('dataset/checkpoint/checkpoint1')

        saver.restore(sess, "dataset/checkpoint/checkpoint1/nn.ckpt-101")







        sess.run(tf.local_variables_initializer())






        for image_path in glob.glob("./dataset/photo/photoresize/*.png"):


            img = Image.open(image_path)

            img = np.array(img)

            print(img.shape)

            if img.ndim == 2:
                img = np.expand_dims(img, 2)
                img = np.concatenate([img, img, img], axis=2)


            #img=np.concatenate([img, img, img], axis=2)

            #image = tf.concat([image, image, image], axis=2)

            img = cv2.resize(img, (448, 448), interpolation=cv2.INTER_CUBIC)

            image = np.expand_dims(img, 0)

            print(image.shape)

            #image = tf.image.resize_bilinear(image, [448, 448])

            print('-')



            img = preprocess_image(image)
            print(img.shape)

            #img = np.concatenate([img, img, img], axis=3)

            #img=np.array(img)






            graph_val = sess.run(box, feed_dict={x: img})


            graph_point = graph_val * 448
            graph_point=np.squeeze(graph_point)
            graph_point=graph_point.tolist()
            graph_points1.append(graph_point)
            print(graph_points1)
        print(graph_points1)


        print('-'*200)

        saver.restore(sess, "dataset/checkpoint/checkpoint2/nn.ckpt-101")

        sess.run(tf.local_variables_initializer())

        for image_path in glob.glob("./dataset/photo/photoresize/*.png"):

            img = Image.open(image_path)

            img = np.array(img)

            print(img.shape)

            if img.ndim == 2:
                img = np.expand_dims(img, 2)
                img = np.concatenate([img, img, img], axis=2)

            # img=np.concatenate([img, img, img], axis=2)

            # image = tf.concat([image, image, image], axis=2)

            img = cv2.resize(img, (448, 448), interpolation=cv2.INTER_CUBIC)

            image = np.expand_dims(img, 0)

            print(image.shape)

            # image = tf.image.resize_bilinear(image, [448, 448])

            print('-')

            img = preprocess_image(image)
            print(img.shape)

            # img = np.concatenate([img, img, img], axis=3)

            # img=np.array(img)

            graph_val = sess.run(box, feed_dict={x: img})

            graph_point = graph_val * 448

            graph_point=np.squeeze(graph_point)
            graph_point=graph_point.tolist()
            graph_points2.append(graph_point)
            print(graph_points2)


    print(graph_points2)
    print(len(graph_points2))

    print(type(graph_points2))



#(x1,y1,x2,y2)  -> (x1,y2,x2,y1) 으로 변형형

    length=len(graph_points1)
    for i in range(length):
        k1=graph_points1[i][3]
        graph_points1[i][3]=graph_points1[i][1]
        graph_points1[i][1]=k1

#
    graph_points3=[]
    for i in range(length):
        graph_points3.append(graph_points1[i][0])
        graph_points3.append(graph_points1[i][1])
        j1=(graph_points1[i][2] + graph_points2[i][0])*0.5
        j2=(graph_points1[i][3] + graph_points2[i][1])*0.5
        graph_points3.append(j1)
        graph_points3.append(j2)
        graph_points3.append(graph_points2[i][2])
        graph_points3.append(graph_points2[i][3])

    for i in range(length):
        graph_points3[i * 6] = graph_points3[i * 6] - graph_points3[i * 6 + 2]
        graph_points3[i * 6 + 1] = graph_points3[i * 6 + 1] - graph_points3[i * 6 + 3]
        graph_points3[i * 6 + 2] = 0
        graph_points3[i * 6 + 3] = 0
        graph_points3[i * 6 + 4] = graph_points3[i * 6 + 4] - graph_points3[i * 6 + 2]
        graph_points3[i * 6 + 5] = graph_points3[i * 6 + 5] - graph_points3[i * 6 + 3]

    adjusted_points=[]
    for i in range(length):
        p1=graph_points3[4:6]
        p2=[0,0]
        p3=graph_points3[i*6+4:i*6+6]
        degree=angle_between(p1, p2, p3)
        print(degree)
        x=graph_points3[i*6]
        y=graph_points3[i*6+1]
        print(x)
        print(y)
        nx,ny=rotation(x,y,degree)
        adjusted_points.append(nx)
        adjusted_points.append(ny)

    print(adjusted_points)

    a1 = adjusted_points[0]
    a2 = adjusted_points[1]
    for i in range(length):
        adjusted_points[i * 2]=adjusted_points[i * 2]-a1
        adjusted_points[i * 2 + 1] = adjusted_points[i * 2 + 1] - a2

    print(adjusted_points)

    adjusted_points_x=[]
    adjusted_points_y=[]
    scatter_pad=[]
    for i in range(length):
        adjusted_points_x.append(adjusted_points[i * 2])
        adjusted_points_y.append(adjusted_points[i * 2 + 1])
        scatter_pad.append(0)
    time=[]
    for i in range(length):
        time.append(i*0.1)

    plt.subplot(221)
    plt.scatter(adjusted_points_x, adjusted_points_y)

    plt.subplot(222)
    plt.plot(adjusted_points_x, adjusted_points_y)

    plt.subplot(223)
    plt.plot(time, adjusted_points_x, 'r', time, adjusted_points_y, 'b')





    plt.show()










if __name__ == '__main__':
    inference()