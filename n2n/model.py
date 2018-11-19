import tensorflow as tf

from .nets import red30


def model_fn(features, labels, mode, params, config):
    is_training = mode == tf.estimator.ModeKeys.TRAIN

    f = tf.make_template('f', red30.model_fn, is_training=is_training)

    x = features['input']
    x_hat = f(x)

    # TODO: Should the output of f be with the final activation? or output logits?

    predictions = None
    if mode in (tf.estimator.ModeKeys.PREDICT, tf.estimator.ModeKeys.EVAL):
        pass

    loss = None
    if mode in (tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL):
        pass

    eval_metric_ops = None
    evaluation_hooks = None
    if mode == tf.estimator.ModeKeys.EVAL:
        pass

    train_op = None
    training_hooks = None
    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            pass

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        training_hooks=training_hooks,
        eval_metric_ops=eval_metric_ops,
        evaluation_hooks=evaluation_hooks)
