# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Training an pixel_cnn model.


"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import json
import os
import time

from absl import app
from absl import flags
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp

from genomics_ood.images_ood import pixel_cnn
from genomics_ood.images_ood import utils

tf.compat.v1.disable_v2_behavior()

flags.DEFINE_string('data_dir', '/tmp/image_data',
                    'Directory to data np arrays.')
flags.DEFINE_string('out_dir', '/tmp/pixelcnn',
                    'Directory to write results and logs.')
flags.DEFINE_boolean('save_im', False, 'If True, save image to npy.')
flags.DEFINE_string('exp', 'fashion', 'cifar or fashion')

# general hyper-parameters
flags.DEFINE_integer('batch_size', 32, 'Batch size for training.')
flags.DEFINE_integer('total_steps', 10, 'Max steps for training')
flags.DEFINE_integer('random_seed', 1234, 'Fixed random seed to use.')
flags.DEFINE_integer('eval_every', 10, 'Interval to evaluate model.')

flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')
flags.DEFINE_float('learning_rate_decay', 0.999995, 'LR decay every step.')

flags.DEFINE_integer('num_logistic_mix', 1,
                     'Number of components in decoder mixture distribution.')
flags.DEFINE_integer('num_hierarchies', 1, 'Number of hierarchies in '
                     'Pixel CNN.')
flags.DEFINE_integer(
    'num_resnet', 5, 'Number of convoluational layers '
    'before depth changes in Pixel CNN.')
flags.DEFINE_integer('num_filters', 32, 'Number of pixel cnn filters')
flags.DEFINE_float('dropout_p', 0.0, 'Dropout probability.')
flags.DEFINE_float('reg_weight', 0.0, 'L2 regularization weight.')
flags.DEFINE_float('mutation_rate', 0.0, 'Mutation rate.')
flags.DEFINE_boolean('use_weight_norm', False,
                     'If True, use weight normalization.')
flags.DEFINE_boolean('data_init', False,
                     ('If True, use data-dependent initialization',
                      ' (has no effect if use_weight_norm is False'))

flags.DEFINE_float('momentum', 0.95, 'Momentum parameter (beta1) for Adam'
                   'optimizer.')
flags.DEFINE_float('momentum2', 0.9995, 'Second momentum parameter (beta2) for'
                   'Adam optimizer.')
flags.DEFINE_boolean('rescale_pixel_value', False,
                     'If True, rescale pixel values into [-1,1].')
flags.DEFINE_boolean('deriv_constraint', False, 'Whether to provide constraint that gradient at f(x) needs to be close to 0.')
flags.DEFINE_float('lambda_penalty', 1.0, 'penalty term for derivative away from 0')
flags.DEFINE_boolean('corr_constraint', False, 'Whether to provide constraint that correlation be 0')
flags.DEFINE_boolean('small_data', False, 'If true, train and validate on 5 data points')
flags.DEFINE_boolean('wasserstein', False, 'If true, train and validate on 5 data points')
flags.DEFINE_integer('wnorm', 1, 'wasserstein norm')
flags.DEFINE_boolean('pixel_hist', False, 'plot tb histograms')
flags.DEFINE_boolean('grad_hist', False, 'plot gradient histograms')
flags.DEFINE_boolean('binarize', False, 'binarize (only for fashion/mnist)')
flags.DEFINE_integer('dist_high', 255, 'max boundary for logistic')
flags.DEFINE_string('dist', 'logistic', 'logistic|categorical|kumaraswamy')
flags.DEFINE_string('output', 'v0', 'version of output scaling')
flags.DEFINE_boolean('condition_count', False, 'plot gradient histograms')
flags.DEFINE_boolean('jsd', False, 'compute jsd instead of kl (i.e. mle)')
flags.DEFINE_float('true_p_off_prob', 1e-8, 'total probability assigned to pixel values other than the true one')
flags.DEFINE_boolean('negative_training', False, 'push down likelihoods for out-dist')
flags.DEFINE_float('neg_log_threshold', -3000, 'log prob threshold')
FLAGS = flags.FLAGS


def main(unused_argv):
  if FLAGS.dist == 'categorical':
    assert FLAGS.binarize, "Must binarize data to use categorical distribution"
  out_dir = FLAGS.out_dir

  exp_dir = 'exp%s' % FLAGS.exp
  model_dir = 'rescale%s' % FLAGS.rescale_pixel_value
  param_dir = 'reg%.2f_mr%.2f' % (FLAGS.reg_weight, FLAGS.mutation_rate)
  job_dir = os.path.join(out_dir, exp_dir, model_dir, param_dir)
  print('job_dir={}'.format(job_dir))

  job_model_dir = os.path.join(job_dir, 'model')
  job_log_dir = os.path.join(job_dir, 'log')

  for sub_dir in out_dir, job_dir, job_model_dir, job_log_dir:
    tf.compat.v1.gfile.MakeDirs(sub_dir)

  if FLAGS.exp in ['fashion', 'mnist', 'ones']:
    n_dim = 28
  elif FLAGS.exp == 'cifar':
    n_dim = 32
  elif FLAGS.exp == 'single_pixel':
    n_dim = 1
  params = {
      'job_model_dir': job_model_dir,
      'job_log_dir': job_log_dir,
      'job_dir': job_dir,
      'dropout_p': FLAGS.dropout_p,
      'reg_weight': FLAGS.reg_weight,
      'num_resnet': FLAGS.num_resnet,
      'num_hierarchies': FLAGS.num_hierarchies,
      'num_filters': FLAGS.num_filters,
      'num_logistic_mix': FLAGS.num_logistic_mix,
      'use_weight_norm': FLAGS.use_weight_norm,
      'data_init': FLAGS.data_init,
      'mutation_rate': FLAGS.mutation_rate,
      'batch_size': FLAGS.batch_size,
      'learning_rate': FLAGS.learning_rate,
      'learning_rate_decay': FLAGS.learning_rate_decay,
      'momentum': FLAGS.momentum,
      'momentum2': FLAGS.momentum2,
      'eval_every': FLAGS.eval_every,
      'save_im': FLAGS.save_im,
      'n_dim': n_dim,
      'n_channel': 1 if FLAGS.exp in ['fashion', 'mnist', 'single_pixel', 'ones'] else 3,
      'exp': FLAGS.exp,
      'rescale_pixel_value': FLAGS.rescale_pixel_value,
      'output': FLAGS.output,
      'condition_count': FLAGS.condition_count,
      'jsd': FLAGS.jsd,
      'true_p_off_prob': FLAGS.true_p_off_prob,
      'negative_training': FLAGS.negative_training,
      'neg_log_threshold': FLAGS.neg_log_threshold
  }

  # Print and write parameter settings
  with tf.io.gfile.GFile(
      os.path.join(params['job_model_dir'], 'params.json'), mode='w') as f:
    f.write(json.dumps(params, sort_keys=True))

  # Fix the random seed - easier to debug separate runs
  tf.compat.v1.set_random_seed(FLAGS.random_seed)

  tf.keras.backend.clear_session()
  sess = tf.compat.v1.Session()
  tf.compat.v1.keras.backend.set_session(sess)
  # Load the datasets
  if FLAGS.exp == 'fashion':
    datasets = utils.load_fmnist_datasets(FLAGS.data_dir, binarize=FLAGS.binarize)
  elif FLAGS.exp == 'mnist':
    datasets = utils.load_mnist_datasets(FLAGS.data_dir, binarize=FLAGS.binarize)
  elif FLAGS.exp == 'ones':
    datasets = utils.load_ones(FLAGS.data_dir)
  elif FLAGS.exp == 'cifar':
    datasets = utils.load_cifar_datasets(FLAGS.data_dir)
  elif FLAGS.exp == 'single_pixel':
    datasets = utils.load_single_pixel_datasets(FLAGS.data_dir)
  
  if FLAGS.small_data:
    datasets['tr_in'] = datasets['tr_in'][:5]
    datasets['val_in'] = datasets['val_in'][:5]

  # pylint: disable=g-long-lambda
  tr_in_ds = datasets['tr_in'].map(lambda x: utils.image_preprocess_add_noise(
      x, params['mutation_rate'])).batch(
          params['batch_size']).repeat().shuffle(1000).make_one_shot_iterator()
  tr_in_im = tr_in_ds.get_next()

  if FLAGS.negative_training:
    tr_ood_ds = datasets['tr_ood'].map(lambda x: utils.image_preprocess_add_noise(
      x, params['mutation_rate'])).batch(
          params['batch_size']).repeat().shuffle(1000).make_one_shot_iterator()
    tr_ood_im = tr_ood_ds.get_next()

  # repeat valid dataset because it will be used for training
  val_in_ds = datasets['val_in'].map(utils.image_preprocess).batch(
      params['batch_size']).repeat().make_one_shot_iterator()
  val_in_im = val_in_ds.get_next()

  # Define a Pixel CNN network
  input_shape = (params['n_dim'], params['n_dim'], params['n_channel'])
  dist = pixel_cnn.PixelCNN(
      image_shape=input_shape,
      dropout_p=params['dropout_p'],
      reg_weight=params['reg_weight'],
      num_resnet=params['num_resnet'],
      num_hierarchies=params['num_hierarchies'],
      num_filters=params['num_filters'],
      num_logistic_mix=params['num_logistic_mix'],
      use_weight_norm=params['use_weight_norm'],
      rescale_pixel_value=params['rescale_pixel_value'],
      high=FLAGS.dist_high,
      output=FLAGS.output,
      conditional_shape=() if FLAGS.condition_count else None
  )

  # Define the training loss and optimizer
  
  if FLAGS.wasserstein:
    # log_prob_i = dist.log_prob(tr_in_im['image'], return_per_pixel=True)  
    mins, maxes = dist.log_prob(tr_in_im['image'], return_per_pixel=True, dist_family='uniform', wasserstein=True)
    if FLAGS.exp == 'fashion':
      mins = tf.squeeze(mins, [-1])  # (B,H,W,1)
      maxes = tf.squeeze(maxes, [-1])
    elif FLAGS.exp == 'cifar':
      pass
    else:
      raise ValueError("Unsupported experiment: ", FLAGS.exp)
    mins = tf.maximum(mins, tf.zeros_like(mins))
    maxes = tf.minimum(maxes, tf.ones_like(maxes) * 255)
    # mins = tf.Print(mins, [tf.reduce_min(mins), tf.reduce_max(mins), tf.reduce_min(maxes), tf.reduce_max(maxes), tf.reduce_min(tr_in_im['image']), tf.reduce_max(tr_in_im['image'])], summarize=10)
    # mins = tf.Print(mins, [tf.shape(mins), tf.shape(maxes), tf.shape(tr_in_im['image'])], "training", summarize=10)
    # mins = tf.Print(mins, [tf.reduce_sum(tf.cast(tf.math.equal(mins, maxes), dtype=tf.int32))], "min equals max", summarize=10)
    mins = tf.Print(mins, [mins, maxes], "min equals max", summarize=10)
    loss = utils.emd(mins, maxes, tr_in_im['image'], FLAGS.lambda_penalty, FLAGS.wnorm)

    val_mins, val_maxes = dist.log_prob(val_in_im['image'], return_per_pixel=True, dist_family='uniform', wasserstein=True)
    val_mins = tf.maximum(val_mins, tf.zeros_like(val_mins))
    val_maxes = tf.minimum(val_maxes, tf.ones_like(val_maxes) * 255)
    if FLAGS.exp == 'fashion':
      val_mins = tf.squeeze(val_mins, [-1])
      val_maxes = tf.squeeze(val_maxes, [-1])
    elif FLAGS.exp == 'cifar':
      pass
    else:
      raise ValueError("Unsupported experiment: ", FLAGS.exp)
    # val_mins = tf.Print(val_mins, [tf.reduce_min(val_mins), tf.reduce_max(val_mins), tf.reduce_min(val_maxes), tf.reduce_max(val_maxes), tf.reduce_min(val_in_im['image']), tf.reduce_max(val_in_im['image'])], summarize=10)
    val_mins = tf.Print(val_mins, [tf.reduce_sum(tf.cast(tf.math.equal(val_mins, val_maxes), dtype=tf.int32))], "min equals max", summarize=10)
    loss_val_in = utils.emd(val_mins, val_maxes, val_in_im['image'], FLAGS.lambda_penalty, FLAGS.wnorm)
  else:
    if FLAGS.condition_count:
      num_zeros = tf.reduce_sum(tf.cast(tf.math.equal(tr_in_im['image'], tf.zeros_like(tr_in_im['image'])), tf.float32), axis=[1, 2, 3]) / 784.
      num_zeros = tf.Print(num_zeros, [tf.shape(num_zeros), tf.shape(tr_in_im['image'])], summarize=10, message="num zeros")
      log_prob_i = dist.log_prob(tr_in_im['image'], return_per_pixel=False, dist_family=FLAGS.dist, conditional_input=num_zeros)
    else:
      log_prob_i = dist.log_prob(tr_in_im['image'], return_per_pixel=False, dist_family=FLAGS.dist)
    # log_prob_i = tf.Print(log_prob_i, [dist.locs, dist.scales], summarize=30, message="train locs and scales")
    # log_prob_i = tf.Print(log_prob_i, [log_prob_i], summarize=30, message="train log probs")
    # log_prob_i = tf.Print(log_prob_i, [tr_in_im['image']], summarize=30, message="train imgs")
    if FLAGS.deriv_constraint:
        img = tf.reshape(tr_in_im['image'], [tf.shape(tr_in_im['image'])[0], -1])
        # proportion of zeros is not differentiable
        fx = tf.reduce_sum(tf.cast(tf.math.equal(img, 0), dtype=tf.int32), axis=1) / tf.shape(img)[1]
        # fx = tf.reduce_sum(img, axis=1)
        grad_px_x = tf.gradients(tf.math.exp(log_prob_i), tr_in_im['image'])[0]
        grad_px_x = tf.reshape(grad_px_x, [tf.shape(grad_px_x)[0], -1])
        tf.print(grad_px_x, output_stream=sys.stdout)
        # grad_fx_x = tf.gradients(fx, img)[0]
        # pseudo-gradient is -1 at 0 and 1 at 1
        grad_fx_x = -1 * tf.cast(tf.math.equal(img, 0), dtype=tf.int32) / tf.shape(img)[1] +  tf.cast(tf.math.greater(img, 0), dtype=tf.int32) / tf.shape(img)[1]
        print(grad_fx_x)
        tf.print(grad_fx_x, output_stream=sys.stdout)
        tf.print(tf.shape(grad_fx_x), output_stream=sys.stdout)
        tf.print(tf.shape(grad_fx_x), output_stream=sys.stdout)
        penalty = FLAGS.lambda_penalty * tf.norm(grad_px_x / tf.cast(grad_fx_x, dtype=tf.float32), axis=1)
        log_prob_i = log_prob_i - penalty
    if FLAGS.corr_constraint:
        img = tf.reshape(tr_in_im['image'], [tf.shape(tr_in_im['image'])[0], -1])
        fx = tf.reduce_sum(tf.cast(tf.math.equal(img, 0), dtype=tf.int32), axis=1) / tf.shape(img)[1]
        corr_px_fx = tfp.stats.correlation(tf.cast(log_prob_i, tf.float32), tf.cast(fx, tf.float32), sample_axis=0, event_axis=None)
        penalty = FLAGS.lambda_penalty * tf.math.abs(corr_px_fx)
        log_prob_i = log_prob_i - penalty
    loss = -tf.reduce_mean(log_prob_i)
    if FLAGS.jsd:
      batch = tf.shape(tr_in_im['image'])[0]
      n_channel = 1 if FLAGS.exp in ['fashion', 'mnist', 'single_pixel', 'ones'] else 3
      values = tf.reshape(tf.repeat(tf.cast(tf.range(256), tf.float32), batch * n_dim * n_dim * n_channel), (256, batch, n_dim, n_dim, n_channel))
      log_qx_i = tf.vectorized_map(dist.learned_log_prob, values)
      # log_qx_i = tf.Print(log_qx_i, [tf.shape(log_qx_i)], message="qx i", summarize=10)
      # place 1 - delta on the pixel value for the image, delta/255 everywhere else
      indices = tf.transpose(tf.unravel_index(indices=[i for i in range(FLAGS.batch_size * n_dim * n_dim * n_channel)], dims=[FLAGS.batch_size, n_dim, n_dim, n_channel]))
      # indices = tf.Print(indices, [tf.shape(indices), tf.shape(tr_in_im['image'])], summarize=10, message='shapes')
      value_and_indices = tf.concat([tf.cast(tf.reshape(tr_in_im['image'], (-1, 1)), tf.int32), indices], axis=1)
      flattened_indices = tf.reduce_sum(value_and_indices, axis=1)
      # TODO probably a more efficient way that doesn't require flattening
      log_px_i = tf.scatter_nd(
        indices=tf.reshape(flattened_indices, (-1, 1)),
        updates=tf.cast(tf.ones_like(flattened_indices), tf.float32) * tf.math.log((1. - 256/255 * FLAGS.true_p_off_prob)),
        shape=tf.constant([FLAGS.batch_size * n_dim * n_dim * n_channel * 256])
      )
      log_px_i += tf.math.log(FLAGS.true_p_off_prob/255.)
      # log_px_i = tf.Print(log_px_i, [tf.shape(log_qx_i), tf.shape(log_px_i)], message="qx px i", summarize=10)
      log_px_i = tf.reshape(log_px_i, (256, FLAGS.batch_size, n_dim, n_dim, n_channel))
      log_qx_i = tf.reshape(log_qx_i, (256, FLAGS.batch_size, n_dim, n_dim, n_channel))
      # log_px_i = tf.reshape(log_px_i, (256, -1))
      # log_qx_i = tf.reshape(log_qx_i, (256, -1))
      penalty = tf.reduce_mean(tf.reduce_sum(tf.math.exp(log_qx_i) * (log_qx_i - log_px_i), axis=[0, 2, 3, 4]))
      loss += penalty
    if FLAGS.negative_training:
      log_prob_i_neg = dist.log_prob(tr_ood_im['image'], return_per_pixel=False, dist_family=FLAGS.dist)
      # indicator = tf.cast(tf.math.greater(log_prob_i_neg, FLAGS.neg_log_threshold), tf.float32)
      # def safe_mean(log_probs, indicator_mask):
      #   x = tf.reduce_sum(indicator_mask)
      #   x_ok = tf.not_equal(x, 0.)
      #   f = lambda x: tf.reduce_sum(log_prob_i_neg * indicator) / x
      #   safe_f = lambda x: 0
      #   safe_x = tf.where(x_ok, x, tf.ones_like(x))
      #   return tf.where(x_ok, f(safe_x), safe_f(x))
      # penalty = safe_mean(log_prob_i_neg, indicator)
      # loss += penalty
      # # loss += tf.reduce_sum(log_prob_i_neg * indicator) / tf.reduce_sum(indicator)
      penalty = tf.reduce_mean(
        tf.where(tf.math.greater(log_prob_i_neg, FLAGS.neg_log_threshold),
          log_prob_i_neg, tf.ones_like(log_prob_i_neg) * FLAGS.neg_log_threshold
      ))
      loss += penalty

    if FLAGS.condition_count:
      num_zeros = tf.reduce_sum(tf.cast(tf.math.equal(val_in_im['image'], tf.zeros_like(val_in_im['image'])), tf.float32), axis=[1, 2, 3])  / 784.
      num_zeros = tf.Print(num_zeros, [tf.shape(num_zeros), tf.shape(val_in_im['image'])], summarize=10, message="num zeros")
      log_prob_i_val_in = dist.log_prob(val_in_im['image'], return_per_pixel=False, dist_family=FLAGS.dist, conditional_input=num_zeros)
    else:
      log_prob_i_val_in = dist.log_prob(val_in_im['image'], dist_family=FLAGS.dist)
    # log_prob_i_val_in = tf.Print(log_prob_i_val_in, [dist.locs, dist.scales], summarize=30, message="val locs and scales")
    # log_prob_i_val_in = tf.Print(log_prob_i_val_in, [log_prob_i_val_in], summarize=30, message="val log probs")
    # log_prob_i_val_in = tf.Print(log_prob_i_val_in, [val_in_im['image']], summarize=30, message="val imgs")
    loss_val_in = -tf.reduce_mean(log_prob_i_val_in)

  global_step = tf.compat.v1.train.get_or_create_global_step()
  learning_rate = tf.compat.v1.train.exponential_decay(
      params['learning_rate'], global_step, 1, params['learning_rate_decay'])
  opt = tf.compat.v1.train.AdamOptimizer(
      learning_rate=learning_rate,
      beta1=params['momentum'],
      beta2=params['momentum2'])
  print('trainable', tf.trainable_variables())
  grads_and_vars=opt.compute_gradients(loss)
  # for g, v in grads_and_vars:
  #     if g is not None:
  #         # g = tf.debugging.check_numerics(g, "{}".format(v.name))
  #         loss = tf.Print(loss, [tf.reduce_sum(tf.cast(tf.math.is_nan(g), tf.int32))], summarize=10, message="{} is nan".format(v.name))
  #         loss = tf.Print(loss, [tf.reduce_min(g), tf.reduce_max(g)], summarize=10, message="{} min and max".format(v.name))
  # grads_and_vars = [(tf.clip_by_value(grad, -1000., 1000.), var) for grad, var in grads_and_vars]
  tr_op = opt.apply_gradients(grads_and_vars)

  # tr_op = opt.minimize(loss, global_step=global_step)

  init_op = tf.compat.v1.global_variables_initializer()
  sess.run(init_op)

  # write tensorflow summaries
  saver = tf.compat.v1.train.Saver(max_to_keep=50000)
  summaries = [
      tf.compat.v1.summary.scalar('loss', loss),
      tf.compat.v1.summary.scalar('train/learning_rate', learning_rate)
  ]
  if FLAGS.pixel_hist:
    locs = dist.locs  # BHWMC
    scales = dist.scales
    # locs = tf.Print(locs, [tf.shape(locs), tf.shape(scales)], summarize=10, message='shape')
    # mixture dim is first
    locs = tf.transpose(locs, perm=[3, 0, 1, 2, 4])
    scales = tf.transpose(scales, perm=[3, 0, 1, 2, 4])
    for idx in range(FLAGS.num_logistic_mix):
      summaries.append(tf.compat.v1.summary.histogram(f'locs_{idx}', locs[idx]))
      summaries.append(tf.compat.v1.summary.histogram(f'scales_{idx}', scales[idx]))
    if FLAGS.exp in ['fashion', 'ones']:
      pixels = tf.expand_dims(tr_in_im['image'], axis=-1)
    elif FLAGS.exp == 'cifar':
      pixels = tr_in_im['image']
    else:
      pixels = tf.ones_like(dist.locs)
    for idx in range(FLAGS.num_logistic_mix):
      summaries.append(tf.compat.v1.summary.histogram(f'loc_{idx}_to_pixel', tf.expand_dims(locs[idx], axis=-1) - pixels))
  if FLAGS.grad_hist:
    for g, v in grads_and_vars:
      if g is not None:
          g = tf.debugging.check_numerics(g, "{}".format(v.name))
          g = tf.Print(g, [g, tf.reduce_sum(tf.cast(tf.math.is_nan(g), tf.int32))], summarize=10, message="{} is nan".format(v.name))
    for g, v in grads_and_vars:
      if g is not None:
          grad_hist_summary = tf.summary.histogram("{}/grad_histogram".format(v.name.replace(':', '_')), g)
          # sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name.replace(':', '_')), tf.nn.zero_fraction(g))
          summaries.append(grad_hist_summary)
          # summaries.append(sparsity_summary)
  if FLAGS.deriv_constraint or FLAGS.corr_constraint or FLAGS.jsd or FLAGS.negative_training:
    summaries.append(tf.compat.v1.summary.scalar('penalty', tf.reduce_mean(penalty)))
  merged_tr = tf.compat.v1.summary.merge(summaries)
  merged_val_in = tf.compat.v1.summary.merge(
      [tf.compat.v1.summary.scalar('loss', loss_val_in)])
  tr_writer = tf.compat.v1.summary.FileWriter(job_log_dir + '/tr_in',
                                              sess.graph)
  val_in_writer = tf.compat.v1.summary.FileWriter(job_log_dir + '/val_in',
                                                  sess.graph)

  # If previous ckpt exists, load ckpt
  ckpt_file = tf.compat.v2.train.latest_checkpoint(job_model_dir)
  if ckpt_file:
    prev_step = int(
        os.path.basename(ckpt_file).split('model_step')[1].split('.ckpt')[0])
    tf.compat.v1.logging.info(
        'previous ckpt exist, prev_step={}'.format(prev_step))
    saver.restore(sess, ckpt_file)
  else:
    prev_step = 0

  # Train the model
  with sess.as_default():  # this is a must otherwise localhost error
    for step in range(prev_step, FLAGS.total_steps + 1, 1):
      # TODO add optimizer back in
      _, loss_tr_np, summary = sess.run([tr_op, loss, merged_tr])
      # loss_tr_np, summary = sess.run([loss, merged_tr])
      if step % params['eval_every'] == 0:
        ckpt_name = 'model_step%d.ckpt' % step
        ckpt_path = os.path.join(job_model_dir, ckpt_name)
        while not tf.compat.v1.gfile.Exists(ckpt_path + '.index'):
          _ = saver.save(sess, ckpt_path, write_meta_graph=False)
          time.sleep(10)
        tr_writer.add_summary(summary, step)

        # Evaluate loss on the valid_in
        loss_val_in_np, summary_val_in = sess.run([loss_val_in, merged_val_in])
        val_in_writer.add_summary(summary_val_in, step)


        # print(step, loss_tr_np, loss_val_in_np)
        print('**************')
        print('step=%d, tr_in_loss=%.4f, val_in_loss=%.4f' %
              (step, loss_tr_np, loss_val_in_np))

        import numpy as np
        if np.isnan(loss_tr_np) or np.isnan(loss_val_in_np):
          import sys; sys.exit()
        tr_writer.flush()
        val_in_writer.flush()

      # if step == 1:  
      #   import sys; sys.exit()

  tr_writer.close()
  val_in_writer.close()


if __name__ == '__main__':
  app.run(main)
