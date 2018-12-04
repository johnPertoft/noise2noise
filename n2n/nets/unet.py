from functools import partial

import tensorflow as tf


def model_fn(x, is_training, output_fn=None):
    conv = partial(tf.layers.conv2d, padding='same')
    pool = partial(tf.layers.max_pooling2d, padding='same')
    activation = partial(tf.nn.leaky_relu, alpha=0.1)

    def concat(a, b):
        return tf.concat((a, b), axis=3)

    def upsample(x):
        # TODO: Not sure if correct. 2x2 is mentioned so should probably explicitly pick nearest neighbor?
        _, h, w, _ = x.shape
        return tf.image.resize_nearest_neighbor(x, (h * 2, w * 2))

    net = x

    net = activation(conv(net, 48, 3, 1))
    net = activation(conv(net, 48, 3, 1))
    net = pool(net, 2, 2)
    pool1 = net

    net = activation(conv(net, 48, 3, 1))
    net = pool(net, 2, 2)
    pool2 = net

    net = activation(conv(net, 48, 3, 1))
    net = pool(net, 2, 2)
    pool3 = net

    net = activation(conv(net, 48, 3, 1))
    net = pool(net, 2, 2)
    pool4 = net

    net = activation(conv(net, 48, 3, 1))
    net = pool(net, 2, 2)

    net = activation(conv(net, 48, 3, 1))
    net = upsample(net)

    net = concat(net, pool4)
    net = activation(conv(net, 96, 3, 1))
    net = activation(conv(net, 96, 3, 1))
    net = upsample(net)

    net = concat(net, pool3)
    net = activation(conv(net, 96, 3, 1))
    net = activation(conv(net, 96, 3, 1))
    net = upsample(net)

    net = concat(net, pool2)
    net = activation(conv(net, 96, 3, 1))
    net = activation(conv(net, 96, 3, 1))
    net = upsample(net)

    net = concat(net, pool1)
    net = activation(conv(net, 96, 3, 1))
    net = activation(conv(net, 96, 3, 1))
    net = upsample(net)

    net = concat(net, x)
    net = activation(conv(net, 64, 3, 1))
    net = activation(conv(net, 32, 3, 1))

    net = conv(net, 3, 3, 1)
    net = output_fn(net)

    return net
