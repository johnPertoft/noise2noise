import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from .dataset import imagenet
from .noise import additive_gaussian_noise
from .noise import text_overlay_noise


#ds = imagenet('/home/john/imagenet-train.tfrecord', 15, noise_fn=additive_gaussian_noise(0, 50))
ds = imagenet('/home/john/imagenet-train.tfrecord', 15, noise_fn=text_overlay_noise(0.0, 0.5))


with tf.Session() as sess:
    img1, img2, ground_truth = ds.make_one_shot_iterator().get_next()
    img1, img2, ground_truth = sess.run((img1, img2, ground_truth))
    _, axes = plt.subplots(5, 3)
    axes = axes.flatten()
    for i1, i2, gt, ax in zip(img1, img2, ground_truth, axes):
        ax.imshow(np.concatenate((i1, i2, gt), axis=1))
    plt.show()
