import tensorflow as tf
import numpy as np

x = tf.placeholder(tf.float32, shape=(9, 9))
y = tf.matmul(x, x)

with tf.Session() as sess:
    rand_array = np.random.rand(9, 9)
    print(sess.run(y, feed_dict={x: rand_array}))