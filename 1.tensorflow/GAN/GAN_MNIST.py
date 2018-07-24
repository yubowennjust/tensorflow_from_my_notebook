import tensorflow as tf
import numpy as np
import datetime
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/")

sample_image = mnist.train.next_batch(1)[0]

print(sample_image.shape)

sample_image = sample_image.reshape([28, 28])
plt.imshow(sample_image, cmap='Greys')
plt.show()

