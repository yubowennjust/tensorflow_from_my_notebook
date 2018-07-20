from datetime import datetime
import math
import time
import tensorflow as tf

def conv_op(input_op,name,kh,kw,n_out,dh,dw,p):
    n_in = input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope+"w",
                                 shape=[kh,kw,n_in,n_out],dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv = tf.nn.conv2d(input_op,kernel,(1,dh,dw,1),padding='SAME')
        bias_init_val = tf.constant(0.0,shape=[n_out],dtype=tf.float32)
        biases = tf.Variable(bias_init_val,trainable=True,name='b')
        z = tf.nn.bias_add(conv,biases)
        activation = tf.nn.relu(z,name=scope)
        p += [kernel,biases]
        return activation

def fc_op(input_op,name,n_out,p):
    n_in = input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope+"w",
                                 shape=[n_in,n_out],dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.Variable(tf.constant(0.1,shape=[n_out],dtype=tf.float32),name='b')
        activation = tf.nn.relu_layer(input_op,kernel,biases,name=scope)
        p +=[kernel,biases]
        return activation

def mpool_op():
    return None
