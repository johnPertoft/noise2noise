from functools import partial

import tensorflow as tf


def model_fn(x, is_training, output_fn=None):
    conv = partial(tf.layers.conv2d, padding='same')
    dconv = partial(tf.layers.conv2d_transpose, padding='same')

    # TODO: Should be relu on sum of current and skip connection?
    # TODO: Not sure about dconv_block.
    # TODO: Try to find a better description of this architecture.

    def conv_block(x, d):
        x = conv(x, d, 3, 1)
        x = tf.nn.relu(x)
        x = tf.layers.batch_normalization(x, training=is_training)
        x = conv(x, d, 3, 2)
        x = tf.nn.relu(x)
        return x

    def dconv_block(x, d):
        x = conv(x, d, 3, 1)
        x = tf.nn.relu(x)
        x = tf.layers.batch_normalization(x, training=is_training)
        x = dconv(x, d, 3, 2)
        x = tf.nn.relu(x)
        return x

    c0 = x
    c1 = conv_block(c0, 128)
    c2 = conv_block(c1, 256)
    c3 = conv_block(c2, 512)

    c4 = conv_block(c3, 1024)
    d5 = dconv_block(c4, 512)

    d6 = dconv_block(d5 + c3, 256)
    d7 = dconv_block(d6 + c2, 128)
    d8 = dconv(d7 + c1, 3, kernel_size=3, strides=2, activation=output_fn)
    x = d8

    return x
