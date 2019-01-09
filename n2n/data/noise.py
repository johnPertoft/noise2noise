import random
import string
from typing import Callable
from typing import Tuple

import cv2
import numpy as np
import tensorflow as tf


NoiseFn = Callable[[tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]


def additive_gaussian_noise(min_stddev: float, max_stddev: float, use_same_distribution: bool = False) -> NoiseFn:
    def apply(img):
        def sample_stddev():
            stddev = tf.random_uniform((tf.shape(img)[0], 1, 1, 1), min_stddev, max_stddev)
            stddev = stddev / 255.0
            return stddev

        stddev1 = sample_stddev()
        stddev2 = sample_stddev() if not use_same_distribution else stddev1

        def add_noise(img, stddev):
            img = img + tf.random_normal(tf.shape(img), stddev=stddev)
            img = tf.clip_by_value(img, 0.0, 1.0)
            return img

        return add_noise(img, stddev1), add_noise(img, stddev2)

    return apply


def brown_additive_gaussian_noise(min_stddev: float, max_stddev: float) -> NoiseFn:
    def apply(img):
        """
        From paper:

        This brown additive noise
        is obtained by blurring white Gaussian noise by a spatial
        Gaussian filter of different bandwidths and scaling to retain
        σ = 25

        Details?? Don't think the following is correct. Doesn't seem to be implemented in official repo.
        * What bandwidths?
        * How to scale to retain sigma = 25?
        """

        stddev = 25.0 / 255
        k = 3
        sigma = 1.0
        coordinates = np.linspace(-k // 2, k // 2, k)
        x, y = np.meshgrid(coordinates, coordinates)
        kernel = np.exp(-(x**2 / (2 * sigma**2) + y ** 2 / (2 * sigma**2)))
        kernel = tf.convert_to_tensor(kernel, dtype=tf.float32)
        kernel = kernel / tf.reduce_sum(kernel)
        kernel = kernel[:, :, tf.newaxis, tf.newaxis]
        kernel = tf.tile(kernel, [1, 1, 3, 1])

        def add_noise(img, stddev):
            noise = tf.random_normal(tf.shape(img), stddev=stddev)
            noise = tf.nn.depthwise_conv2d(noise, kernel, [1, 1, 1, 1], padding='SAME')
            img = img + noise
            img = tf.clip_by_value(img, 0.0, 1.0)
            return img

        return add_noise(img, stddev), add_noise(img, stddev)

    return apply


def bernoulli_noise(min_corruption_rate: float,
                    max_corruption_rate: float,
                    use_same_distribution: bool = False) -> NoiseFn:
    def apply(img):
        def sample_corruption_rate():
            return tf.random.uniform((tf.shape(img)[0],), minval=min_corruption_rate, maxval=max_corruption_rate)

        p1 = sample_corruption_rate()
        p2 = sample_corruption_rate() if not use_same_distribution else p1

        # TODO: Need to pass the gradient masks somehow to not backpropagate the gradients
        # from missing pixels in targets?

        def apply_mask(img, p):
            probs = p[:, tf.newaxis, tf.newaxis] * tf.ones(tf.shape(img)[:3])
            mask = tf.distributions.Bernoulli(probs=probs).sample()
            mask = tf.cast(mask, tf.float32)
            mask = mask[..., tf.newaxis]
            mask = tf.tile(mask, (1, 1, 1, 3))
            img = img * mask
            return img

        return apply_mask(img, p1), apply_mask(img, p2)

    return apply


def poisson_noise():
    def apply(img):
        pass

    return apply


def text_overlay_noise(min_coverage: float, max_coverage: float, use_same_distribution: bool = False) -> NoiseFn:
    def apply(img):
        def sample_coverage():
            return tf.random_uniform([tf.shape(img)[0]], min_coverage, max_coverage)

        coverage1 = sample_coverage()
        coverage2 = sample_coverage() if not use_same_distribution else coverage1

        def add_text_py(img, coverage):
            h, w, _ = img.shape

            img = img * 255.0
            img = img.astype(np.uint8)

            text_overlay = np.zeros((h, w, 4), img.dtype)

            while text_overlay[:, :, 3].sum() < h * w * coverage:
                font = cv2.FONT_HERSHEY_SIMPLEX
                text = ''.join([random.choice(string.ascii_letters + string.digits)
                                for _ in range(np.random.randint(5, 15))])
                font_scale = np.random.uniform(0.5, 1.0)
                thickness = np.random.randint(1, 3)

                # Note: Setting alpha channel to 1 to use for counting how many pixels are covered.
                color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256), 1)

                (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

                # TODO: Fix this, slightly off for text in the bottom of the image.
                # Note: allowing text to lie slightly outside image to ensure that pixels close to edges have the
                # same/similar probability of being corrupted.
                p = 5
                x = np.random.randint(-p, max(-p + 1, w - tw + p))
                y = np.random.randint(th - p, max(th - p + 1, h - th + p))

                cv2.putText(
                    text_overlay,
                    text,
                    (x, y),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=font_scale,
                    color=color,
                    thickness=thickness,
                    lineType=cv2.LINE_AA)

            overlay_pixels = text_overlay[:, :, 3] > 0
            img[overlay_pixels] = text_overlay[overlay_pixels, :3]

            img = img / 255.0
            img = img.astype(np.float32)

            return img

        def add_text(img, coverage):
            _, h, w, c = img.get_shape().as_list()

            def add_text_single(img, coverage):
                img = tf.py_func(add_text_py, [img, coverage], tf.float32)
                img = tf.ensure_shape(img, (h, w, c))
                return img

            img = tf.map_fn(lambda args: add_text_single(*args), [img, coverage], dtype=tf.float32)
            return img

        return add_text(img, coverage1), add_text(img, coverage2)

    return apply


def impulse_noise():
    def apply(img):
        pass

    return apply
