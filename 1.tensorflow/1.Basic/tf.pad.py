import tensorflow as tf
import numpy as np

if __name__ == "__main__":
    a = np.array(np.arange(1, 10))
    a = a.reshape((3, 3))
    print(a)
    sess = tf.InteractiveSession()
    b = tf.pad(a, [[1, 1], [2, 2]])
    print(sess.run(b))
    b = tf.pad(a,[[1,1],[2,2]],"REFLECT")
    print(sess.run(b))
    b = tf.pad(a,[[1,1],[2,2]],"SYMMETRIC")
    print(sess.run(b))

    a = np.array(np.arange(1,9))
    a = a.reshape((2,2,2))
    print(a)
    sess = tf.InteractiveSession()
    b = tf.pad(a,[[1,1],[2,2],[3,3]],constant_values=0)
    print(sess.run(b))

    a = np.array(np.arange(1, 9))
    a = a.reshape((2, 2, 2))
    print(a)
    sess = tf.InteractiveSession()
    b = tf.pad(a, [[1, 1], [2, 2], [3, 3]], constant_values=0)
    print(sess.run(b))
