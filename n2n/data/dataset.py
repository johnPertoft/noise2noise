from typing import Optional

import tensorflow as tf

from .noise import NoiseFn


def imagenet(file_pattern: str,
             batch_size: int,
             seed: Optional[int] = None) -> tf.data.Dataset:

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

    return ds


def noisy_imagenet(file_pattern: str,
                   batch_size: int,
                   noise_fn: NoiseFn,
                   seed: Optional[int] = None) -> tf.data.Dataset:

    def apply_noise(img):
        a, b = noise_fn(img)
        return a, b, img

    ds = imagenet(file_pattern, batch_size, seed)
    ds = ds.map(apply_noise)
    ds = ds.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)

    return ds
