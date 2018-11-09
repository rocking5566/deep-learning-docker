from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data as mnist_data


# Convenience method for reshaping images. The included MNIST dataset stores images
# as Nx784 row vectors. This method reshapes the inputs into Nx28x28x1 images that are
# better suited for convolution operations and rescales the inputs so they have a
# mean of 0 and unit variance.
def resize_images(images, mean, std):
    reshaped = (images - mean) / std
    reshaped = np.reshape(reshaped, [-1, 28, 28, 1])

    assert (reshaped.shape[1] == 28)
    assert (reshaped.shape[2] == 28)
    assert (reshaped.shape[3] == 1)

    return reshaped


def nielsen_net(inputs, is_training, scope='NielsenNet'):
    with tf.variable_scope(scope, 'NielsenNet'):
        # First Group: Convolution + Pooling 28x28x1 => 28x28x20 => 14x14x20
        net = slim.conv2d(inputs, 20, [5, 5], padding='SAME', scope='layer1-conv')
        net = slim.max_pool2d(net, 2, stride=2, scope='layer2-max-pool')

        # Second Group: Convolution + Pooling 14x14x20 => 10x10x40 => 5x5x40
        net = slim.conv2d(net, 40, [5, 5], padding='VALID', scope='layer3-conv')
        net = slim.max_pool2d(net, 2, stride=2, scope='layer4-max-pool')

        # Reshape: 5x5x40 => 1000x1
        net = tf.reshape(net, [-1, 5*5*40])

        # Fully Connected Layer: 1000x1 => 1000x1
        net = slim.fully_connected(net, 1000, scope='layer5')
        net = slim.dropout(net, is_training=is_training, scope='layer5-dropout')

        # Second Fully Connected: 1000x1 => 1000x1
        net = slim.fully_connected(net, 1000, scope='layer6')
        net = slim.dropout(net, is_training=is_training, scope='layer6-dropout')

        # Output Layer: 1000x1 => 10x1
        net = slim.fully_connected(net, 10, scope='output')
        net = slim.dropout(net, is_training=is_training, scope='output-dropout')

        return net


def main():
    # Read in MNIST dataset, compute mean / standard deviation of the training images
    mnist = mnist_data.read_data_sets('MNIST_data', one_hot=True)

    mean = np.mean(mnist.train.images)
    std = np.std(mnist.train.images)

    # Create the placeholder tensors for the input images (x), the training labels (y_actual)
    # and whether or not dropout is active (is_training)
    x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='Inputs')
    y_actual = tf.placeholder(tf.float32, shape=[None, 10], name='Labels')
    is_training = tf.placeholder(tf.bool, name='IsTraining')

    # Pass the inputs into nielsen_net, outputting the logits
    logits = nielsen_net(x, is_training, scope='NielsenNetTrain')

    # Use the logits to create four additional operations:
    #
    # 1: The cross entropy of the predictions vs. the actual labels
    # 2: The number of correct predictions
    # 3: The accuracy given the number of correct predictions
    # 4: The update step, using the MomentumOptimizer
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_actual))
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_actual, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    train_step = tf.train.MomentumOptimizer(0.01, 0.5).minimize(cross_entropy)

    # To monitor our progress using tensorboard, create two summary operations
    # to track the loss and the accuracy
    loss_summary = tf.summary.scalar('loss', cross_entropy)
    accuracy_summary = tf.summary.scalar('accuracy', accuracy)

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter('/log/tensorboard', sess.graph)

        eval_data = {
            x: resize_images(mnist.validation.images, mean, std),
            y_actual: mnist.validation.labels,
            is_training: False
        }

        for i in range(1000):
            images, labels = mnist.train.next_batch(128)
            summary, _ = sess.run([loss_summary, train_step],
                                  feed_dict={x: resize_images(images, mean, std), y_actual: labels, is_training: True})
            train_writer.add_summary(summary, i)

            if i % 50 == 0:
                summary, acc = sess.run([accuracy_summary, accuracy], feed_dict=eval_data)
                train_writer.add_summary(summary, i)
                print("Step: %5d, Validation Accuracy = %5.2f%%" % (i, acc * 100))

        test_data = {
            x: resize_images(mnist.test.images, mean, std),
            y_actual: mnist.test.labels,
            is_training: False
        }

        acc = sess.run(accuracy, feed_dict=test_data)

        print("Test Accuracy = %5.2f%%" % (100 * acc))


if __name__ == '__main__':
    main()
