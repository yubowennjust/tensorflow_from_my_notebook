import tensorflow as tf

v = tf.Variable(0,dtype=tf.float32,name="v")
# v1 = tf.Variable(1,dtype=tf.float32,name="v1")

for variables in tf.global_variables():
    print(variables.name)

ema = tf.train.ExponentialMovingAverage(0.99)

maintain_average_op = ema.apply(tf.global_variables())

for variables in tf.global_variables():
    print(variables.name)

saver = tf.train.Saver()

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    sess.run(tf.assign(v,10))
    sess.run(maintain_average_op)

    saver.save(sess,"../model/model3.ckpt")
    print(sess.run([v,ema.average(v)]))

v = tf.Variable(0, dtype=tf.float32, name="v")
saver = tf.train.Saver({"v/ExponentialMovingAverage":v})
with tf.Session() as sess:
    saver.restore(sess,"../model/model3.ckpt")
    print(sess.run(v))



