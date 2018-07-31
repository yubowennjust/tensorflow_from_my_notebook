import tensorflow as tf


v1 = tf.get_variable("v", [1])
print(v1.name)

with tf.variable_scope("foo"):
    v2 = tf.get_variable("v", [1])
print(v2.name)

with tf.variable_scope("foo"):
    with tf.variable_scope("bar"):
        v3 = tf.get_variable("v", [1])
        print(v3.name)

v4 = tf.get_variable("v1", [1])
print(v4.name)