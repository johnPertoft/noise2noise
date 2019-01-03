import tensorflow as tf

from .data import imagenet
from .data.noise import additive_gaussian_noise
from .data.noise import bernoulli_noise
from .data.noise import text_overlay_noise
from .model import model_fn


tf.app.flags.DEFINE_string('model_dir', None, 'Where to place model checkpoint files.')
tf.app.flags.DEFINE_string('train_files', None, 'File pattern of tfrecord files for training (string).')
tf.app.flags.DEFINE_string('eval_files', None, 'File pattern of tfrecord files for evaluation (string).')
tf.app.flags.DEFINE_integer('batch_size', 8, 'The batch size.')
tf.app.flags.DEFINE_integer('eval_batch_size', 128, 'The batch size for evaluation.')
tf.app.flags.DEFINE_integer('num_epochs', 100, 'Number of epochs.')

tf.app.flags.DEFINE_string('noise', 'additive_gaussian', 'The noise type to add.')

tf.app.flags.mark_flags_as_required(['train_files', 'eval_files'])

FLAGS = tf.app.flags.FLAGS


def main(argv):
    # TODO: Need conditional parameters for each noise type as well as for train and eval.
    if FLAGS.noise == 'additive_gaussian':
        train_noise = additive_gaussian_noise(0, 50)
        eval_noise = additive_gaussian_noise(25, 25)
    elif FLAGS.noise == 'bernoulli':
        train_noise = bernoulli_noise(0.0, 0.95)
        eval_noise = bernoulli_noise(0.5, 0.5)
        # TODO: Should account for the masked gradients.
    elif FLAGS.noise == 'text':
        train_noise = text_overlay_noise(0, 0.5)
        eval_noise = text_overlay_noise(0.25, 0.25)
    else:
        raise ValueError(f'Invalid noise: {FLAGS.noise}.')

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=FLAGS.model_dir)

    def input_fn_train():
        ds = imagenet(FLAGS.train_files, FLAGS.batch_size, train_noise)
        ds = ds.map(lambda img1, img2, gt: {'input': img1, 'target': img2, 'gt': gt})
        return ds

    def input_fn_eval():
        ds = imagenet(FLAGS.eval_files, FLAGS.eval_batch_size, eval_noise)
        ds = ds.map(lambda img1, img2, gt: {'input': img1, 'target': img2, 'gt': gt})
        return ds

    train_spec = tf.estimator.TrainSpec(
        input_fn=input_fn_train)

    eval_spec = tf.estimator.EvalSpec(
        input_fn=input_fn_eval,
        start_delay_secs=120,
        throttle_secs=600)

    # TODO: Add option of using clean targets for comparison.

    # TODO: Experiments with adversarial losses?

    # TODO: Doesn't seem to reinit dataset. Just do explicit train/eval loop instead?
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    # TODO: Test set evaluation.

    # TODO: Show comparisons with simpler techniques.
        # Simple average of the two noisy realizations.


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
