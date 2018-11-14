import tensorflow as tf


def model_fn(features, labels, mode, params, config):

    # TODO: Use tf gan estimator?

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        training_hooks=training_hooks,
        eval_metric_ops=eval_metric_ops,
        evaluation_hooks=evaluation_hooks)