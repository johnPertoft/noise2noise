import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from .dataset import imagenet
from .noise import additive_gaussian_noise


ds = imagenet('/home/john/imagenet.tfrecord', 15, noise_fn=additive_gaussian_noise(0, 50))
img1, img2, ground_truth = ds.make_one_shot_iterator().get_next()

with tf.Session() as sess:
    img1, img2, ground_truth = sess.run((img1, img2, ground_truth))
    _, axes = plt.subplots(5, 3)
    axes = axes.flatten()
    for i1, i2, gt, ax in zip(img1, img2, ground_truth, axes):
        ax.imshow(np.concatenate((i1, i2, gt), axis=1))
    plt.show()
