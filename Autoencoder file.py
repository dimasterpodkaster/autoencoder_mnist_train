import tensorflow as tf
import keras
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def plot_image(image):
    fig = plt.figure()
    plt.imshow(image, cmap="Greys_r")
    plt.axis("off")
    plt.show()
    print(image.shape)


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
print(x_train.shape)

# plot_image(x_train[0])

n_inputs = 28 * 28
BATCH_SIZE = 1

tf.compat.v1.disable_eager_execution()
batch_size = tf.compat.v1.placeholder(tf.int64)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, n_inputs])

dataset = tf.compat.v1.data.Dataset.from_tensor_slices(x).repeat().batch(batch_size)
iter = tf.compat.v1.data.make_initializable_iterator(dataset)
features = iter.get_next()

# Normalization of input images
data_from_dataset = np.array(x_train)
data_from_dataset = data_from_dataset.reshape(data_from_dataset.shape[0], -1)
print(data_from_dataset.shape, data_from_dataset[0])

with tf.compat.v1.Session() as sess:
    sess.run(iter.initializer, feed_dict={x: data_from_dataset, batch_size: BATCH_SIZE})
    print(sess.run(features).shape)
    plt.imshow(sess.run(features).reshape([28, 28]), cmap="Greys_r")
    plt.show()

data_from_dataset = data_from_dataset.astype('float32') / 255.
