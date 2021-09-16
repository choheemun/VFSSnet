import numpy as np
import tensorflow as tf

'''
gt_boxes=np.array([[1,2,3,4,5],[2,3,4,5,6],[3,4,5,6,7]])
gt_box=gt_boxes[:,:4]
print(gt_box)

gt_box=gt_box/10
print(gt_box)
'''
''''
QUEUE_LENGTH = 10

q = tf.FIFOQueue(QUEUE_LENGTH,"float")
enq_ops = q.enqueue_many(([1.0,2.0,3.0,4.0],) )
qr = tf.train.QueueRunner(q,[enq_ops,enq_ops,enq_ops])

sess = tf.Session()

# Create a coordinator, launch the queue runner threads.
coord = tf.train.Coordinator()
threads = qr.create_threads(sess, coord=coord, start=True)



for step in range(20):
    print(sess.run(q.dequeue()))



coord.request_stop()
coord.join(threads)
sess.close()



q=tf.train.shuffle_batch([], batch_size=10, capacity=100, min_after_dequeue=10)
'''

'''
import tensorflow as tf
from PIL import Image
image = Image.open("./p/vfss1.png")

img_array = np.array(image)
height, width, channel = img_array.shape
print(height, width , channel)


image=img_array[:,:,:,np.newaxis]
a = np.random.randint(380, 449)
b = np.random.randint(380, 449)
image = tf.image.resize_bilinear(image, [a, b])

im = np.zeros((448, 448, 3), dtype=np.uint8)

im[:, :, :] = image[:, :, :]
image = im
image.show()


img=np.array([[1,2],[3,4]])
im = np.zeros((3, 3), dtype=np.uint8)
im[:(1+1), :2] = img[:, :]
img=im
print(img)



a=np.array([1,2,3,4])
b=a*(10,10,10,10)
c=b+(1,1,1,1)
print(c)
'''
image = tf.ones([3,3,1])
a=tf.concat([image,image,image],axis=0)

#a=tf.expand_dims(image,)

sess = tf.Session()
print(sess.run(a))



