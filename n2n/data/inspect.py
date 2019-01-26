import numpy as np
import cv2
import tensorflow as tf

from .dataset import noisy_imagenet
from .noise import additive_gaussian_noise
from .noise import bernoulli_noise
from .noise import brown_additive_gaussian_noise
from .noise import impulse_noise
from .noise import text_overlay_noise


tf.app.flags.DEFINE_string('files', None, 'File pattern of tfrecord files for inspection.')
tf.app.flags.DEFINE_string('noise', 'additive_gaussian', 'The noise type to add.')
tf.app.flags.mark_flags_as_required(['files'])
FLAGS = tf.app.flags.FLAGS


def main(argv):
    if FLAGS.noise == 'additive_gaussian':
        noise_fn = additive_gaussian_noise(0, 50)
    elif FLAGS.noise == 'bernoulli':
        noise_fn = bernoulli_noise(0.0, 0.95)
    elif FLAGS.noise == 'brown_additive_gaussian':
        noise_fn = brown_additive_gaussian_noise(0, 50)
    elif FLAGS.noise == 'impulse':
        noise_fn = impulse_noise(0.0, 0.95)
    elif FLAGS.noise == 'text':
        noise_fn = text_overlay_noise(0.0, 0.5)
    else:
        raise ValueError(f'Invalid noise: {FLAGS.noise}.')

    ds = noisy_imagenet(FLAGS.files, 15, noise_fn=noise_fn)

    with tf.Session() as sess:
        img1, img2, ground_truth = ds.make_one_shot_iterator().get_next()
        img1, img2, ground_truth = sess.run((img1, img2, ground_truth))

        img = np.concatenate((img1, img2, ground_truth), axis=2)
        img = np.concatenate(img, axis=0)
        img = (img * 255).astype(np.uint8)
        img = img[:, :, [2, 1, 0]]  # cv2 expects BGR.
        cv2.imwrite('noise-inspect-out.png', img)


if __name__ == '__main__':
    tf.app.run(main)
