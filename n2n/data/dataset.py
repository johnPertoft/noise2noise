import tensorflow as tf


# TODO: Noise fns
    # salt and pepper / random color noise
    # text overlay noise
    # short/long exposure, possible?


def gaussian_noise(img):

    # TODO: Define a mask with some pixels set to some random value.

    # TODO: Or use scatter_nd?

    # Add noise and clip pixel values.
    pass


def text_overlay_noise(img):
    # Probably implement with py_func
    pass


def imagenet(file_pattern: str, batch_size: int, seed=None) -> tf.data.Dataset:

    def parse_fn(example):
        parsed = tf.parse_single_example(example, {'image/encoded': tf.FixedLenFeature((), tf.string)})

        image = tf.image.decode_jpeg(parsed['image/encoded'], channels=3)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.image.resize_images(image, (224, 224), align_corners=False, method=tf.image.ResizeMethod.BILINEAR)

        return image

    files = tf.data.Dataset.list_files(file_pattern, seed=seed)
    ds = files.interleave(tf.data.TFRecordDataset, cycle_length=8)
    ds = ds.shuffle(1024, seed=seed)
    ds = ds.map(parse_fn)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)

    return ds
