import tensorflow as tf

from .nets import rednet


def model_fn(features, mode):
    is_training = mode == tf.estimator.ModeKeys.TRAIN

    f = tf.make_template('f', rednet.model_fn, is_training=is_training)

    # TODO: Preprocess x to [-1, 1]? or keep output of f in [0, 1]?

    x_hat = f(features['input'])

    # TODO: Arguments for choosing loss.
    loss_fn = tf.losses.mean_squared_error

    loss = None
    if mode in (tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL):
        loss = loss_fn(x_hat, features['target'])

    train_op = None
    training_hooks = None
    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_op = tf.train.AdamOptimizer(1e-4).minimize(loss, global_step=global_step)

    eval_metric_ops = None
    evaluation_hooks = None
    if mode == tf.estimator.ModeKeys.EVAL:
        # TODO: Add psnr
        eval_metric_ops = {
            'gt_loss': tf.metrics.mean(loss_fn(f(features['input']), features['gt']))
        }

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=x_hat,
        loss=loss,
        train_op=train_op,
        training_hooks=training_hooks,
        eval_metric_ops=eval_metric_ops,
        evaluation_hooks=evaluation_hooks)
