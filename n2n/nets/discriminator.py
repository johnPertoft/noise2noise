import functools

import tensorflow as tf


def model_fn(x, is_training):

    # CycleGAN inspired patch-gan discriminator.

    activation = functools.partial(tf.nn.leaky_relu, alpha=0.2)
    conv = functools.partial(tf.layers.conv2d, padding='same')
    norm = tf.contrib.layers.instance_norm

    net = x
    net = activation(conv(net, 32, 4, 2))
    net = activation(norm(conv(net, 64, 4, 2)))
    net = activation(norm(conv(net, 128, 4, 2)))
    net = activation(norm(conv(net, 256, 4, 2)))
    net = activation(norm(conv(net, 512, 4, 1)))
    net = conv(net, 1, 4, 1)

    return net

