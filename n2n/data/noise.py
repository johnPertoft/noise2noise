import tensorflow as tf

# TODO: Brown noise.
# TODO: text overlay noise.


def additive_gaussian_noise(min_stddev, max_stddev):
    def apply(img):
        n = tf.shape(img)[0]
        stddev = tf.random_uniform((n, 1, 1, 1), min_stddev, max_stddev)
        stddev = stddev / 255.0
        clip = lambda i: tf.clip_by_value(i, 0.0, 1.0)
        noisy_img1 = clip(img + tf.random_normal(tf.shape(img), stddev=stddev))
        noisy_img2 = clip(img + tf.random_normal(tf.shape(img), stddev=stddev))
        return noisy_img1, noisy_img2

    return apply


def text_overlay_noise():
    def apply(img):
        pass

    return apply
