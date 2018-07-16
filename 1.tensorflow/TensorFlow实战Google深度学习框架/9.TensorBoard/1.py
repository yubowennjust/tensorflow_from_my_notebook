# -*- coding: utf-8 -*-
# @Time    : 2018/07/16 13:49
# @Author  : 于博文
# @Email   : yubowen.njust@gmail.com
# @File    : 1.py
# @Software: PyCharm

import tensorflow as tf

with tf.name_scope("input1"):
    input1 = tf.constant([1.0, 2.0, 3.0], name="input2")
with tf.name_scope("input2"):
    input2 = tf.Variable(tf.random_uniform([3]), name="input2")
output = tf.add_n([input1, input2], name="add")

writer = tf.summary.FileWriter("log/simple_example.log", tf.get_default_graph())

writer.close()