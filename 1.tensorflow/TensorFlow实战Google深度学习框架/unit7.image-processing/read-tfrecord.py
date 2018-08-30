import tensorflow as tf

reader = tf.TFRecordReader()
filename_queue = tf.train.string_input_producer(["../output.tfrecords"])
_,serialized_example = reader.read(filename_queue)

# 解析读取的样例。
features = tf.parse_single_example(
    serialized_example,
    features={
        'image_raw':tf.FixedLenFeature([],tf.string),
        'pixels':tf.FixedLenFeature([],tf.int64),
        'label':tf.FixedLenFeature([],tf.int64)
    })

images = tf.decode_raw(features['image_raw'],tf.uint8)
labels = tf.cast(features['label'],tf.int32)
pixels = tf.cast(features['pixels'],tf.int32)

sess = tf.Session()

# 启动多线程处理输入数据。
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess,coord=coord)
import pylab

for i in range(10):
    image, label, pixel = sess.run([images, labels, pixels])
    #
    # im = mnist.train.images[0]
    # im = images.reshape(-1, 28)



    print(image)

    print("11111111111111111111111111111111111111111111111111111111111111111111111111111")