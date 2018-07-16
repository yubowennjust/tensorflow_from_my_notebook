# -*- coding: utf-8 -*-
# @Time    : 2018/07/16 8:36
# @Author  : 于博文
# @Email   : yubowen.njust@gmail.com
# @File    : 1.py
# @Software: PyCharm

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

image_raw_data = tf.gfile.FastGFile("1.jpg",'rb').read()

with tf.Session() as sess:

    img_data = tf.image.decode_jpeg(image_raw_data)

    print(img_data.eval())

    img_data.set_shape([1797,2673,3])

    print(img_data.get_shape())

with tf.Session() as sess:

    plt.imshow(img_data.eval())

    plt.show()

with tf.Session() as sess:

    resized = tf.image.resize_images(img_data, [300, 300], method=0)

    # TensorFlow的函数处理图片后存储的数据是float32格式的，需要转换成uint8才能正确打印图片。
    print
    "Digital type: ", resized.dtype
    cat = np.asarray(resized.eval(), dtype='uint8')
    # tf.image.convert_image_dtype(rgb_image, tf.float32)
    plt.imshow(cat)
    plt.show()

with tf.Session() as sess:
    croped = tf.image.resize_image_with_crop_or_pad(img_data, 1000, 1000)
    # padded = tf.image.resize_image_with_crop_or_pad(img_data, 3000, 3000)
    plt.imshow(croped.eval())
    plt.show()
    # plt.imshow(padded.eval())
    # plt.show()

with tf.Session() as sess:
    central_cropped = tf.image.central_crop(img_data, 0.5)
    plt.imshow(central_cropped.eval())
    plt.show()

with tf.Session() as sess:
    # 上下翻转
    # flipped1 = tf.image.flip_up_down(img_data)
    # 左右翻转
    # flipped2 = tf.image.flip_left_right(img_data)

    # 对角线翻转
    transposed = tf.image.transpose_image(img_data)
    plt.imshow(transposed.eval())
    plt.show()

    # 以一定概率上下翻转图片。
    # flipped = tf.image.random_flip_up_down(img_data)
    # 以一定概率左右翻转图片。
    # flipped = tf.image.random_flip_left_right(img_data)

with tf.Session() as sess:
    # 将图片的亮度-0.5。
    # adjusted = tf.image.adjust_brightness(img_data, -0.5)

    # 将图片的亮度-0.5
    # adjusted = tf.image.adjust_brightness(img_data, 0.5)

    # 在[-max_delta, max_delta)的范围随机调整图片的亮度。
    adjusted = tf.image.random_brightness(img_data, max_delta=0.5)

    # 将图片的对比度-5
    # adjusted = tf.image.adjust_contrast(img_data, -5)

    # 将图片的对比度+5
    # adjusted = tf.image.adjust_contrast(img_data, 5)

    # 在[lower, upper]的范围随机调整图的对比度。
    # adjusted = tf.image.random_contrast(img_data, lower, upper)

    plt.imshow(adjusted.eval())
    plt.show()

with tf.Session() as sess:
    adjusted = tf.image.adjust_hue(img_data, 0.5)
    # adjusted = tf.image.adjust_hue(img_data, 0.3)
    # adjusted = tf.image.adjust_hue(img_data, 0.6)
    # adjusted = tf.image.adjust_hue(img_data, 0.9)

    # 在[-max_delta, max_delta]的范围随机调整图片的色相。max_delta的取值在[0, 0.5]之间。
    # adjusted = tf.image.random_hue(image, max_delta)

    # 将图片的饱和度-5。
    # adjusted = tf.image.adjust_saturation(img_data, -5)
    # 将图片的饱和度+5。
    # adjusted = tf.image.adjust_saturation(img_data, 5)
    # 在[lower, upper]的范围随机调整图的饱和度。
    # adjusted = tf.image.random_saturation(img_data, lower, upper)

    # 将代表一张图片的三维矩阵中的数字均值变为0，方差变为1。
    # adjusted = tf.image.per_image_whitening(img_data)

    plt.imshow(adjusted.eval())
    plt.show()
