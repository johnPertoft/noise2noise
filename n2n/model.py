import tensorflow as tf

from .nets import discriminator
from .nets import rednet
from .nets import unet


tf.app.flags.DEFINE_string('architecture', 'unet', 'The network architecture to use.')
tf.app.flags.DEFINE_string('loss', 'l2', 'The loss function to use.')
tf.app.flags.DEFINE_string('adv_loss', None, 'The adversarial loss to use.')
tf.app.flags.DEFINE_float('adv_loss_weight', 1.0, 'The weight of the adversarial loss in the total loss.')
tf.app.flags.DEFINE_float('learning_rate', 1e-4, 'The learning rate.')
tf.app.flags.DEFINE_boolean('variable_histograms', False, 'Whether to add histogram summaries for model variables.')
tf.app.flags.DEFINE_boolean('gradient_histograms', False, 'Whether to add histogram summaries for model gradients.')

FLAGS = tf.app.flags.FLAGS


# TODO: Add PSNR metric.
# TODO: Paper mentions rampdown period but no details. Check reference implementation.
# TODO: Add a comparison of some simple image processing approach. I.e. median of neighborhood or similar.
# TODO: Add summary of average metric of noise img vs ground truth as well for comparison.


def model_fn(features, labels, mode, config):
    assert labels is None, '`labels` argument should not be used.'

    is_training = mode == tf.estimator.ModeKeys.TRAIN
    global_step = tf.train.get_or_create_global_step()

    if FLAGS.architecture == 'unet':
        architecture = unet.model_fn
    elif FLAGS.architecture == 'rednet':
        architecture = rednet.model_fn
    else:
        raise ValueError(f'Invalid architecture: `{FLAGS.architecture}`.')

    if FLAGS.loss == 'l0':
        def l0_loss(labels, predictions):
            max_steps = 200_000
            ratio = tf.math.minimum(global_step, max_steps) / max_steps
            gamma = 2 * (1 - ratio)
            gamma = tf.cast(gamma, tf.float32)
            loss = (tf.abs(labels - predictions) + 1e-8) ** gamma
            loss = tf.reduce_mean(loss)
            return loss
        loss_fn = l0_loss
    elif FLAGS.loss == 'l1':
        loss_fn = tf.losses.absolute_difference
    elif FLAGS.loss == 'l2':
        loss_fn = tf.losses.mean_squared_error
    else:
        raise ValueError(f'Invalid loss: `{FLAGS.loss}`.')

    denoise = tf.make_template('denoise', architecture, is_training=is_training, output_fn=tf.nn.sigmoid)

    x_hat = denoise(features['input'])

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=x_hat,
            export_outputs={'denoised': x_hat})

    loss = loss_fn(x_hat, features['target'])

    if FLAGS.adv_loss is not None:
        d_loss, adv_loss = add_adversarial_loss(x_hat, is_training)
        loss = loss + FLAGS.adv_loss_weight * adv_loss

    if mode == tf.estimator.ModeKeys.EVAL:
        tf.summary.image('denoising', tf.concat((features['input'], x_hat, features['gt']), axis=2))

        crop_central_fraction = 0.4
        crop = tf.concat((
            tf.image.central_crop(features['input'], crop_central_fraction),
            tf.image.central_crop(x_hat, crop_central_fraction),
            tf.image.central_crop(features['gt'], crop_central_fraction)),
            axis=2)
        _, ch, cw, _ = crop.shape.as_list()
        crop = tf.image.resize_images(crop, (int(ch * 1.5), int(cw * 1.5)))
        tf.summary.image('denoising_zoomed', crop)

        eval_summary_hook = tf.train.SummarySaverHook(
            save_steps=1,
            output_dir=f'{config.model_dir}/eval',
            summary_op=tf.summary.merge_all())

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            eval_metric_ops={'ground_truth_loss': tf.metrics.mean(loss_fn(x_hat, features['gt']))},
            evaluation_hooks=[eval_summary_hook])

    if mode == tf.estimator.ModeKeys.TRAIN:
        learning_rate = FLAGS.learning_rate
        tf.summary.scalar('learning_rate', learning_rate)

        optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate,
            beta1=0.9,
            beta2=0.999)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            grads_and_vars = optimizer.compute_gradients(loss, var_list=tf.trainable_variables('denoise'))
            train_op = optimizer.apply_gradients(grads_and_vars, global_step)

        for g, v in grads_and_vars:
            if FLAGS.variable_histograms:
                tf.summary.histogram(v.op.name, v)
            if FLAGS.gradient_histograms:
                tf.summary.histogram(g.op.name, g)

        if FLAGS.adv_loss is not None:
            d_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5)
            d_train_op = d_optimizer.minimize(d_loss, var_list=tf.trainable_variables('discriminate'))
            train_op = tf.group(train_op, d_train_op)

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op)


def add_adversarial_loss(x_fake, is_training):
    # TODO: Refactor this.
    from .data import imagenet
    clean_ds = imagenet(FLAGS.train_files, FLAGS.batch_size).repeat(None)
    x_real = clean_ds.make_one_shot_iterator().get_next()

    discriminate = tf.make_template('discriminate', discriminator.model_fn, is_training=is_training)

    d_real = discriminate(x_real)
    d_fake = discriminate(x_fake)

    if FLAGS.adv_loss == 'lsgan':
        d_loss = tf.losses.mean_squared_error(tf.ones_like(d_real), d_real) + \
                 tf.losses.mean_squared_error(tf.zeros_like(d_fake), d_fake)
        d_loss = d_loss / 2.0

        g_loss = tf.losses.mean_squared_error(tf.ones_like(d_fake), d_fake)
        g_loss = g_loss / 2.0
    else:
        raise ValueError(f'Invalid adversarial loss `{FLAGS.adv_loss}`')

    tf.summary.scalar('adv_d_loss', d_loss)
    tf.summary.scalar('adv_g_loss', g_loss)

    return d_loss, g_loss
