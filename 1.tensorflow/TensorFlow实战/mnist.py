# -*- coding: utf-8 -*-
# @Time    : 2018/07/16 21:43
# @Author  : 于博文
# @Email   : yubowen.njust@gmail.com
# @File    : mnist.py
# @Software: PyCharm

import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print(mnist.train.images.shape,mnist.train.labels.shape)

print(mnist.test.images.shape,mnist.test.labels.shape)

print(mnist.validation.images.shape,mnist.validation.labels.shape)

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32,[None,784])

w = tf.Variable(tf.zeros([784,10]))

b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x,w) + b)

y_ = tf.placeholder(tf.float32,[None,10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

tf.global_variables_initializer().run()

for i in range(1000):
    batch_xs,batch_ys = mnist.train.next_batch(1000)
    train_step.run({x:batch_xs,y_:batch_ys})
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print(i,accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))

