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




# libs/datasets/dataset_factory.py


def preprocess_image(image, is_training=False):
    """preprocess image for coco
    1. random flipping
    2. min size resizing
    3. zero mean          ----- 이것만
    4. ...
    """

    image = tf.cast(image, tf.float32)
    image = image / 256.0
    image = (image - 0.5) * 2.0
    #image = tf.expand_dims(image, axis=0)     # batch 때문에 늘리는듯


    return image









def root_mean_squared_error(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.losses.mean_squared_error(y_true, y_pred)))


def inference():
    """The main function that runs training"""

    ## data


    files = (glob.glob("./dataset/photo/photoresize/*.png"))
    length=len(files)
    q = tf.train.string_input_producer(files,shuffle=False)
    reader = tf.WholeFileReader()
    file_name, content = reader.read(q)
    image = tf.image.decode_jpeg(content, channels=1)
    image = tf.cast(image, tf.float32)
    print(image.shape)
    image = tf.concat([image, image, image], axis=2)
    print(image.shape)
    image=tf.expand_dims(image,0)


    print(image.shape)
    image = tf.image.resize_bilinear(image, [448, 448])
    print(image.shape)
    image = preprocess_image(image)
    print(image.shape)





    '''
    W = tf.random.uniform([2], 190, 224)
    W = tf.round(W)
    a=W[0]
    b=W[1]
    image=tf.image.resize_bilinear(image, [2*a, 2*b])
    padding=[[0,0],[224-a,224-a],[224-b,224-b],[0,0]]
    image = tf.pad(image, padding,mode="CONSTANT")
    gt_box=gt_box*(2*a/448,2*b/448,2*a/448,2*b/448)
    gt_box=gt_box + (224-a,224-b,224-a,224-b)

    gt_box = gt_box / 200  # w=200, h=200  추후에 변경
    '''






    model = ResNet50(input_tensor=image, include_top=False, weights='imagenet', pooling='max')
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
    #loss = root_mean_squared_error(gt_box, box)  # (None, 4)  vs  (None, 4)
    #train_step = tf.train.AdamOptimizer(learning_rate=0.000001).minimize(loss, global_step=global_step)

    sess = tf.Session()
    k.set_session(sess)

    graph_points1 = []
    graph_points2 = []
    graph_points3 = []


    with tf.Session() as sess:
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=20)
        #ckpt = tf.train.get_checkpoint_state('dataset/checkpoint/checkpoint1')

        saver.restore(sess, "dataset/checkpoint/checkpoint1/nn.ckpt-4401")


        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # sess.run(tf.global_variables_initializer())
        for i in range(length):
            if coord.should_stop():
                break
            sess.run(tf.local_variables_initializer())

            graph_point = sess.run(box)
            graph_point = graph_point * 448
            graph_points1.append(graph_point)



        coord.request_stop()
        coord.join(threads)

    print(graph_points1)

    '''
    with tf.Session() as sess:

        # option 1
        # init = tf.global_variables_initializer()
        # sess.run(init)
        # init = tf.local_variables_initializer()
        # sess.run(init)
        #saver = tf.train.Saver(tf.global_variables(), max_to_keep=20)     optimize 하지 않고, 파라메타를 저장도X
        ckpt = tf.train.get_checkpoint_state('dataset/checkpoint/checkpoint1')
        #if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        #saver.restore(sess, ckpt.model_checkpoint_path)  # 1.단순히 epoch를 연장하여 학습을 이어가는 경우
        saver.restore(sess, "./checkpoint/nn.ckpt-3500")              # 2.over fitting 이 확인되어 특정 epoch를 선택하여 학습을 이어가는 경우
        # 1 or 2 중 하나만 실행시키면 된다.
        sess.run(tf.local_variables_initializer())  # string_input_producer 의 epoch 때문...

        #else:
        #    sess.run(tf.global_variables_initializer())
        #    sess.run(tf.local_variables_initializer())

        with tf.Session() as sess:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            #sess.run(tf.global_variables_initializer())
            computed_weights = sess.run(box)
            # computed_weights: (5, 30250) array

            coord.request_stop()
            coord.join(threads)
    



        
        coord = tf.train.Coordinator()
        threads = []
        # print (tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS))
        for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
            threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                             start=True))

        tf.train.start_queue_runners(sess=sess, coord=coord)

        for step in range(10000):
            sess.run(train_step)  # feed_dict를 쓰지 않는다. input pipeline 이 data를 feed 해준다
            if step % 100 == 0:
                loss_val = sess.run(loss)
                print('Step: {:4d} | Loss: {:.5f}'.format(step, loss_val))
                #print(sess.run([gt_box, box,W]))
                print(sess.run([box]))
                print('-')
                checkpoint_path = os.path.join('dataset/checkpoint/checkpoint1', 'nn.ckpt')
                saver.save(sess, checkpoint_path, global_step=global_step)
            if coord.should_stop():
                coord.request_stop()
                coord.join(threads)
    '''






if __name__ == '__main__':
    inference()