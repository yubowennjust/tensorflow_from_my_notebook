# -*- coding: utf-8 -*-
# @Time    : 2018/07/11 20:32
# @Author  : 于博文
# @Email   : yubowen.njust@gmail.com
# @File    : python_save.py
# @Software: PyCharm

import tensorflow as tf

# v1 = tf.Variable(tf.constant(1.0,shape=[1]),name='v1')
# v2 = tf.Variable(tf.constant(2.0,shape=[1]),name='v2')
#
# result = v1 + v2
#
# init_op = tf.global_variables_initializer()
#
# saver = tf.train.Saver()
#
# with tf.Session() as sess:
#     sess.run(init_op)
#     saver.save(sess,"model/model.ckpt")




# v1 = tf.Variable(tf.constant(1.0,shape=[1]),name='v1')
# v2 = tf.Variable(tf.constant(2.0,shape=[1]),name='v2')
#
# result = v1 + v2
#
# saver = tf.train.Saver()
#
# with tf.Session() as sess:
#     saver.restore(sess,"model/model.ckpt")
#     print(sess.run(result))




# v = tf.Variable(0,dtype=tf.float32,name='v')
# for variables in tf.global_variables():
#     print(variables.name)
#
# ema = tf.train.ExponentialMovingAverage(0.99)
# maintain_average_op = ema.apply(tf.global_variables())
#
# for variables in tf.global_variables():
#     print(variables.name)
#
# saver = tf.train.Saver()
#
# with tf.Session() as sess:
#     init_op = tf.global_variables_initializer()
#     sess.run(init_op)
#
#     sess.run(tf.assign(v,10))
#     sess.run(maintain_average_op)
#     saver.save(sess, "model/model.ckpt")
#     print(sess.run([v,ema.average(v)]))




# v = tf.Variable(0,dtype=tf.float32,name='v')
# saver = tf.train.Saver({"v/ExponentialMovingAverage":v})
# with tf.Session() as sess:
#     saver.restore(sess,"model/model.ckpt")
#     print(sess.run(v))



# v = tf.Variable(0,dtype=tf.float32,name='v')
# ema = tf.train.ExponentialMovingAverage(0.99)
# print(ema.variables_to_restore())
#
# saver = tf.train.Saver(ema.variables_to_restore())
# with tf.Session() as sess:
#     saver.restore(sess, "model/model.ckpt")
#     print(sess.run(v))




# from tensorflow.python.framework import graph_util
#
# v1 = tf.Variable(tf.constant(1.0,shape=[1]),name='v1')
# v2 = tf.Variable(tf.constant(2.0,shape=[1]),name='v2')
# result = v1 + v2
#
# init_op = tf.initialize_all_variables()
# with tf.Session() as sess:
#     sess.run(init_op)
#     graph_def = tf.get_default_graph().as_graph_def()
#     output_graph_def = graph_util.convert_variables_to_constants(sess,graph_def,['add'])
#
#     with tf.gfile.GFile("model/combined_model.pb","wb") as f:
#         f.write(output_graph_def.SerializeToString())



# from tensorflow.python.platform import gfile
# with tf.Session() as sess:
#     model_filename = "model/combined_model.pb"
#     with gfile.FastGFile(model_filename,'rb') as f:
#         graph_def = tf.GraphDef()
#         graph_def.ParseFromString(f.read())
#     result = tf.import_graph_def(graph_def,return_elements=["add:0"])
#     print(sess.run(result))



v1 = tf.Variable(tf.constant(1.0,shape=[1]),name='v1')
v2 = tf.Variable(tf.constant(2.0,shape=[1]),name='v2')
result = v1 + v2
saver = tf.train.Saver()

saver.export_meta_graph("to/model.ckpt.meda.json",as_text=True)