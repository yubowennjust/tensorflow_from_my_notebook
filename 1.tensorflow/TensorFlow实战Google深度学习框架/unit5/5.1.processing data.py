from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../unit5/MNIST_data",one_hot=True)

print("training data size",mnist.train.num_examples)

print("validation data size",mnist.validation.num_examples)

print("training test size",mnist.test.num_examples)

print("example data size",mnist.train.images[0])

print("example data label",mnist.train.labels[0])

import pylab

im = mnist.train.images[0]
im = im.reshape(-1,28)
pylab.imshow(im)
pylab.show()

batch_size = 100
xs,ys = mnist.train.next_batch(batch_size)
print("X shape",xs.shape)
print("Y shape",ys.shape)