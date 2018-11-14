import tensorflow as tf

from .dataset import imagenet

ds = imagenet('/efs/datasets/imagenet/train*', 64)

img = ds.make_one_shot_iterator().get_next()
with tf.Session() as sess:
    print(sess.run(img)[0])
