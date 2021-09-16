'''
from __future__ import print_function
import tensorflow as tf
import time
import numpy as np


# 세션을 실행한다.
sess = tf.InteractiveSession()

# 사이즈 100 큐를 생성하고 eqneue 노드를 정의한다.
# 임의의 값을 enqueue하는 enqueue_op 노드를 정의한다.
gen_random_normal = tf.random_normal(shape=())
queue = tf.train.shuffle_batch([gen_random_normal], batch_size=10, capacity=1000, min_after_dequeue=10)

# 10개의 쓰레드를 만들고 각각의 쓰레드가 병렬로(parallel) enqueue_op operation을 비동기적으로(asynchronous) 실행한다.
# 쓰레드를 컨트롤 할 수 있는 tf.train.Coordinator를 선언하고 각각의 쓰레드들을 tf.train.Coordinator에 넣어준다.

coord = tf.train.Coordinator()
threads=tf.train.start_queue_runners(sess=sess, coord=coord)
q=queue*10
print(queue)
# 10개의 쓰레드가 병렬적으로 연산을 수행한다.
# 아웃풋 예시 :
# 25
# 77
# 100

print(sess.run(tf.shape(queue)))
print(sess.run(queue))
print(sess.run(q))

coord.request_stop()
coord.join(threads)

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
from PIL import Image
import cv2



'''
a=np.array([[1,2,3,4]])
b=np.array([[1,2,3,4,5,6]])
#image=tf.cast(a,tf.float32)


x = tf.placeholder(tf.float32, shape=(None, 4))
labels=tf.placeholder(tf.float32, shape=(None, 6))

m = Dense(20, activation='relu', kernel_initializer='he_normal')(x)
m=m/10
m = Dense(200, activation='relu', kernel_initializer='he_normal')(m)
m=m/10
m = Dense(200, activation='relu', kernel_initializer='he_normal')(m)
m=m/10
m = Dense(200, activation='relu', kernel_initializer='he_normal')(m)
m=m/10
pred = Dense(6, activation='sigmoid', kernel_initializer='he_normal')(m)
pred=pred*6






loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(labels, pred))
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

sess=tf.Session()
k.set_session(sess)

global_step = tf.Variable(0, trainable=False, name='global_step')

init_op = tf.global_variables_initializer()
sess.run(init_op)

saver = tf.train.Saver(tf.global_variables(), max_to_keep=20)
ckpt = tf.train.get_checkpoint_state('dataset/checkpoint/checkpoint12')

for i in range(10000):
    _, cost=sess.run([train_step,loss], feed_dict={x: a, labels: b})
    if i%1000==0:
        print(i)
        print(cost)

checkpoint_path = os.path.join('dataset/checkpoint/checkpoint12', 'nn.ckpt')
saver.save(sess, checkpoint_path, global_step=global_step)

sess.close()



a=np.array([1,2,3])
print(a)
print(type(a))

print('-'*50)

b=[1,2,3,4,5,6]
c=[[1,2,3],[4,5,6]]
print(type(b))
print(type(c))
print(len(b))
print(len(c))

print('-'*50)

print(c)

a=[1,2,3,4]
k1=(a[0]+a[1])*0.5
k2=(a[2]+a[3])*0.5
print(k1)
print(k2)
a.append(k1)
a.append(k1)

print(a)
num=2
print(a[num*2])
a[num*2]=a[num*2]-a[num-2]
print(a)
print(a[num*2+1])
print('-'*50)
print(a[num:num+2])
print('-'*50)


from math import atan2, degrees
import math
def angle_between(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    deg1 = (360 + degrees(atan2(x1 - x2, y1 - y2))) % 360
    deg2 = (360 + degrees(atan2(x3 - x2, y3 - y2))) % 360
    return deg2 - deg1 if deg1 <= deg2 else 360 - (deg1 - deg2)


p1=[0,-1]
p2=[0,0]
p3=[1,-1]

print(angle_between(p1,p2,p3))




from math import atan2, degrees
import math

def rotation(x,y,degree):
    rad = degree * (math.pi / 180.0)
    nx = (math.cos(rad) * x - math.sin(rad) * y)
    ny = (math.sin(rad) * x + math.cos(rad) * y)
    return nx, ny

print(rotation(100.12,24.72,180))
'''
import matplotlib.pyplot as plt

adjusted_points_x = [4.85,4.81,4.76,4.68,4.59,4.31,3.94,3.70,3.54,3.69,3.92,4.13,4.35,4.52,4.71,4.78,4.88,4.90,4.79,4.92,]
adjusted_points_y = [5.76,5.39,5.22,5.06,4.67,4.39,3.66,3.26,3.05,3.06,3.32,3.44,3.52,3.75,3.89,4.14,4.62,4.98,5.31,5.74,]
scatter_pad = []



adjusted_points_x2=[]
for i in range(len(adjusted_points_x)):
    k=adjusted_points_x[i]
    k=10-k
    adjusted_points_x2.append(k)


adjusted_points_y2=[]
for i in range(len(adjusted_points_y)):
    j=adjusted_points_y[i]
    j=10-j
    adjusted_points_y2.append(j)




time = []
for i in range(20):
    k=i/20
    time.append(k)

print(time)
plt.subplot(221)
plt.scatter(adjusted_points_x, adjusted_points_y2)

plt.subplot(222)
plt.plot(adjusted_points_x, adjusted_points_y2)

plt.subplot(223)
plt.plot(time, adjusted_points_x2, 'r', time, adjusted_points_y2, 'b')

plt.show()