# -*- coding: utf-8 -*-
# @Time    : 2018/07/09 14:52
# @Author  : 于博文
# @Email   : yubowen.njust@gmail.com
# @File    : unit3.py
# @Software: PyCharm

# # 3.1
# import tensorflow as tf
#
# a = tf.constant([1.0, 2.0], name="a")
# b = tf.constant([2.0, 3.0], name="b")
#
# result = a + b
# sess = tf.Session()
# sess.run(result)
#
# print(sess.run(result))
#
# print(a.graph is tf.get_default_graph())
#
# g1 = tf.Graph()
#
# with g1.as_default():
#     v = tf.get_variable(
#         "v", shape=[1], initializer=tf.zeros_initializer()
#     )
#
# g2 = tf.Graph()
#
# with g2.as_default():
#     v = tf.get_variable(
#         "v", shape=[1], initializer=tf.ones_initializer()
#     )
#
# with tf.Session(graph=g1) as sess:
#     tf.global_variables_initializer().run()
#     with tf.variable_scope("", reuse=True):
#         print(sess.run(tf.get_variable("v")))
#
# with tf.Session(graph=g2) as sess:
#     tf.global_variables_initializer().run()
#     with tf.variable_scope("", reuse=True):
#         print(sess.run(tf.get_variable("v")))
#
# g = tf.Graph()
# with g.device('/gpu:0'):
#     result = a+b
#
#     sess = tf.Session()
#     sess.run(result)
#
#     print(sess.run(result))
#
#
# # 3.2
# print("++++++++++++++++++++++++3.2+++++++++++++++++++++++++++++++")
# a = tf.constant([1.0, 2.0], name="a")
# b = tf.constant([2.0, 3.0], name="b")
# result = tf.add(a,b,name="add")
# print(result)
#
# # 3.3
# print("++++++++++++++++++++++++3.3+++++++++++++++++++++++++++++++")
# a = tf.constant([1.0, 2.0], name="a")
# b = tf.constant([2.0, 3.0], name="b")
#
# result = a + b
# sess = tf.Session()
# sess.run(result)
#
# print(sess.run(result))
# print(result.eval(session=sess))
#
#
# #3.4
# print("++++++++++++++++++++++++3.4+++++++++++++++++++++++++++++++")
# w1 = tf.Variable(tf.random_normal([2,3], stddev=1.0, seed=1))
# w2 = tf.Variable(tf.random_normal([3,1], stddev=1.0, seed=1))
#
# x = tf.constant([[0.7,0.9]])
#
# a = tf.matmul(x,w1)
# y = tf.matmul(a,w2)
#
# sess = tf.Session()
#
# sess.run(w1.initializer)
# sess.run(w2.initializer)
#
# print(sess.run(y))
# sess.close()
#
#
# print("++++++++++++++++++++++++placeholder+++++++++++++++++++++++++++++++")
# w1 = tf.Variable(tf.random_normal([2,3], stddev=1.0))
# w2 = tf.Variable(tf.random_normal([3,1], stddev=1.0))
#
# x = tf.placeholder(tf.float32,shape=(1,2),name="input")
#
# a = tf.matmul(x,w1)
# y = tf.matmul(a,w2)
#
# sess = tf.Session()
# init_op = tf.global_variables_initializer()
# sess.run(init_op)
#
# print(sess.run(y,feed_dict={x:[[0.7,0.9]]}))


# all code ini

import tensorflow as tf
import numpy as np
from numpy.random import RandomState

batch_size = 8

w1 = tf.Variable(tf.random_normal([2,3], stddev=1.0, seed=1))
w2 = tf.Variable(tf.random_normal([3,1], stddev=1.0, seed=1))

x = tf.placeholder(tf.float32, shape=(None,2),name='x-input')
y_ = tf.placeholder(tf.float32, shape=(None,1),name='y-input')

a = tf.matmul(x,w1)
y = tf.matmul(a,w2)

cross_entropy =  -tf.reduce_mean(
    y_*tf.log(tf.clip_by_value(y,1e-10,1.0))
)

train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

rdm = RandomState(1)

dataset_size = 128

X = rdm.rand(dataset_size,2)

Y = [[int(x1+x2<1)] for (x1,x2) in X]

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print(sess.run(w1))
    print(sess.run(w2))

    STEPS = 5000
    for i in range(STEPS):
        start = (i*batch_size)%dataset_size
        end = min(start+batch_size,dataset_size)

        sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})

        if i%1000==0:
            total_cross_entropy = sess.run(
                cross_entropy,feed_dict={x:X,y_:Y}
            )
            print("after %d training step(s) cross entropy on all data is %g"%(i,total_cross_entropy))
    print(sess.run(w1))
    print(sess.run(w2))