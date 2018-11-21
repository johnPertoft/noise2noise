import tensorflow as tf

from .data import imagenet
from .data.noise import additive_gaussian_noise
from .model import model_fn


tf.app.flags.DEFINE_string('model_dir', None, 'Where to place model checkpoint files.')
tf.app.flags.DEFINE_string('train_files', None, 'File pattern of tfrecord files for training.')
tf.app.flags.DEFINE_string('eval_files', None, 'File pattern of tfrecord files for evaluation.')
tf.app.flags.DEFINE_integer('batch_size', 64, 'The batch size.')
tf.app.flags.DEFINE_integer('eval_batch_size', 128, 'The batch size for evaluation.')
tf.app.flags.DEFINE_integer('num_epochs', 100, 'Number of epochs.')

tf.app.flags.mark_flags_as_required(['train_files', 'eval_files'])

FLAGS = tf.app.flags.FLAGS


def main(argv):
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=FLAGS.model_dir)

    def input_fn_train():
        ds = imagenet(FLAGS.train_files, FLAGS.batch_size, additive_gaussian_noise(0, 50))
        ds = ds.map(lambda img1, img2, gt: {'input': img1, 'target': img2, 'gt': gt})
        return ds

    def input_fn_eval():
        ds = imagenet(FLAGS.eval_files, FLAGS.eval_batch_size, additive_gaussian_noise(25, 25))
        ds = ds.map(lambda img1, img2, gt: {'input': img1, 'target': img2, 'gt': gt})
        return ds

    # TODO: Add summaries.

    train_spec = tf.estimator.TrainSpec(
        input_fn=input_fn_train,
        hooks=[])

    eval_spec = tf.estimator.EvalSpec(
        input_fn=input_fn_eval,
        start_delay_secs=120,
        throttle_secs=600,
        hooks=[])

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    # TODO: Test set evaluation.


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
