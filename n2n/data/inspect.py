import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from .dataset import imagenet
from .noise import additive_gaussian_noise
from .noise import brown_additive_gaussian_noise
from .noise import text_overlay_noise
from .noise import bernoulli_noise


tf.app.flags.DEFINE_string('files', None, 'File pattern of tfrecord files for inspection.')
tf.app.flags.DEFINE_string('noise', 'additive_gaussian', 'The noise type to add.')
tf.app.flags.mark_flags_as_required(['files'])
FLAGS = tf.app.flags.FLAGS


def main(argv):
    if FLAGS.noise == 'additive_gaussian':
        noise_fn = additive_gaussian_noise(0, 50)
    elif FLAGS.noise == 'brown_additive_gaussian':
        noise_fn = brown_additive_gaussian_noise(0, 50)
    elif FLAGS.noise == 'text':
        noise_fn = text_overlay_noise(0.0, 0.5)
    elif FLAGS.noise == 'bernoulli':
        noise_fn = bernoulli_noise(0.0, 0.95)
    else:
        raise ValueError(f'Invalid noise: {FLAGS.noise}.')

    ds = imagenet(FLAGS.files, 15, noise_fn=noise_fn)

    with tf.Session() as sess:
        img1, img2, ground_truth = ds.make_one_shot_iterator().get_next()
        img1, img2, ground_truth = sess.run((img1, img2, ground_truth))
        _, axes = plt.subplots(5, 3)
        axes = axes.flatten()
        for i1, i2, gt, ax in zip(img1, img2, ground_truth, axes):
            ax.imshow(np.concatenate((i1, i2, gt), axis=1))
        plt.show()


if __name__ == '__main__':
    tf.app.run(main)
