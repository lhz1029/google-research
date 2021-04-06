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

r"""Evaluating Likelihood Ratios based on pixel_cnn model.


"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import flags
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow.compat.v1 as tf
from matplotlib.lines import Line2D
import csv

from genomics_ood.images_ood import pixel_cnn
from genomics_ood.images_ood import utils

from scipy.stats import wasserstein_distance
import ot

tf.compat.v1.disable_v2_behavior()

flags.DEFINE_string('model_dir', '/tmp/expfashion/rescaleFalse/',
                    'Directory to write results and logs.')
flags.DEFINE_string('data_dir', '/misc/vlgscratch5/RanganathGroup/lily/ood/data',
                    'Directory to data np arrays.')
flags.DEFINE_integer('ckpt_step', 10, 'The step of the selected ckpt.')
flags.DEFINE_string('exp', 'fashion-mnist', 'describe both in and out-dist')
flags.DEFINE_integer(
    'repeat_id', -1,
    ('We run 10 independent experiments to get the mean and variance of AUROC.',
     'repeat_id=i indicates the i-th independent run.',
     'repeat_id=-1 indecates only one independent run.'))
flags.DEFINE_boolean('color_classes', False, 'whether to color by class (only for fashion mnist and mnist)')
flags.DEFINE_string('dist_family', 'logistic', 'logistic|uniform|categorical')
flags.DEFINE_boolean('binarize', False, 'whether to binarize data')
FLAGS = flags.FLAGS

REG_WEIGHT_LIST = [0, 10, 100]
MUTATION_RATE_LIST = [0.1, 0.2, 0.3]

output_file = "ood_eval_emd_03_10_21.csv"

def load_datasets(exp, data_dir):
  if exp == 'fashion-mnist':
    datasets = utils.load_fmnist_datasets(data_dir, binarize=FLAGS.binarize)
  elif exp == 'fashion-vflip':
    datasets = utils.load_fmnist_datasets(data_dir, out_data='vflip')
  elif exp == 'fashion-hflip':
    datasets = utils.load_fmnist_datasets(data_dir, out_data='hflip')
  elif exp == 'fashion-omniglot':
    datasets = utils.load_fmnist_datasets(data_dir, out_data='omniglot')
  elif exp == 'fashion-gaussian':  # TODO need to deal with negatives
    datasets = utils.load_fmnist_datasets(data_dir, out_data='gaussian')
  elif exp == 'fashion-uniform':  # TODO unif is uniform dist, but it's sometimes actually "constant" in the lit
    datasets = utils.load_fmnist_datasets(data_dir, out_data='unif')
  elif exp == 'mnist-fashion':
    datasets = utils.load_mnist_datasets(data_dir)
  elif exp == 'cifar-svhn':
    datasets = utils.load_cifar_datasets(data_dir)
  elif exp == 'cifar-celeba':
    datasets = utils.load_cifar_datasets(data_dir, out_data='celeba')
  elif exp == 'cifar-vflip':
    datasets = utils.load_cifar_datasets(data_dir, out_data='vflip')
  elif exp == 'cifar-hflip':
    datasets = utils.load_cifar_datasets(data_dir, out_data='hflip')
  elif exp == 'cifar-gaussian':
    datasets = utils.load_cifar_datasets(data_dir, out_data='gaussian')
  elif exp == 'cifar-uniform':
    datasets = utils.load_cifar_datasets(data_dir, out_data='unif')
  elif exp == 'cifar-cifar100':
    datasets = utils.load_cifar_datasets(data_dir, out_data='cifar100')
  elif exp == 'cifar-imagenet32':
    datasets = utils.load_cifar_datasets(data_dir, out_data='imagenet32')
  return datasets


def find_ckpt_match_param(reg_weight, mutation_rate, repeat_id, ckpt_step):
  """Find model ckpt that is trained based on mutation_rate and reg_weight."""
  param_dir = 'reg%.2f_mr%.2f' % (reg_weight, mutation_rate)
  ckpt_dir = os.path.join(FLAGS.model_dir, param_dir)
  print(ckpt_dir)
  if repeat_id == -1:
    ckpt_repeat_dir = os.path.join(ckpt_dir, 'model')
  else:
    # each param_dir may have multiple independent runs
    try:
      repeat_dir_list = tf.compat.v1.gfile.ListDirectory(ckpt_dir)
    except tf.errors.NotFoundError:
      return None
    repeat_dir = repeat_dir_list[repeat_id]
    ckpt_repeat_dir = os.path.join(ckpt_dir, repeat_dir) # , 'model')
  print(ckpt_repeat_dir)
  ckpt_file = utils.get_ckpt_at_step(ckpt_repeat_dir, ckpt_step)
  print('ckpt_file={}'.format(ckpt_file))
  return ckpt_file


def create_model_and_restore_ckpt(ckpt_file):
  """Restore model from ckpt."""
  # load params
  params_json_file = os.path.join(os.path.dirname(ckpt_file), 'params.json')
  params = utils.load_hparams(params_json_file)

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
      output=params.get('output', 'v0'),
      conditional_shape=() if params.get('condition_count', False) else None,
  )

  saver = tf.compat.v1.train.Saver(max_to_keep=50000)
  init_op = tf.compat.v1.global_variables_initializer()

  # restore ckpt
  sess = tf.compat.v1.Session()
  tf.compat.v1.keras.backend.set_session(sess)
  sess.run(init_op)
  saver.restore(sess, ckpt_file)

  return dist, params, sess


def load_data_and_model_and_pred(exp,
                                 data_dir,
                                 reg_weight,
                                 mutation_rate,
                                 repeat_id,
                                 ckpt_step,
                                 eval_mode_in,
                                 eval_mode_ood,
                                 return_per_pixel=False):
  """Load datasets, load model ckpt, and eval the model on the datasets."""
  tf.compat.v1.reset_default_graph()
  # load datasets
  datasets = load_datasets(exp, data_dir)
  # load model
  ckpt_file = find_ckpt_match_param(reg_weight, mutation_rate, repeat_id,
                                    ckpt_step)
  print('CKPT: ', ckpt_file)
  if not ckpt_file:  # no ckpt file is found
    # with open(output_file, 'a') as f:
    #   f.write('{}: bad model\n'.format(FLAGS.model_dir))
    raise ValueError('No ckpt model found')
    return None, None, None, None

  dist, params, sess = create_model_and_restore_ckpt(ckpt_file)
  condition_count = params.get('condition_count', False)

  # if condition_count:
  #   samples = dist.sample((1, 672), dist_family=FLAGS.dist_family, conditional_input=list(range(59, 731)))
  # else:
  #   samples = dist.sample(500, dist_family=FLAGS.dist_family)
  # samples_np, = sess.run([samples])
  # print(samples_np.shape)
  # np.save(f'{FLAGS.model_dir}/samples', samples_np)
  # import sys; sys.exit()

  # Evaluations
  preds_in = utils.eval_on_data(
      datasets['%s_in' % eval_mode_in],
      utils.image_preprocess,
      params,
      dist,
      sess,
      return_per_pixel=return_per_pixel,
      dist_family=FLAGS.dist_family,
      wasserstein=True,
      condition_count=condition_count)
  preds_ood = utils.eval_on_data(
      datasets['%s_ood' % eval_mode_ood],
      utils.image_preprocess,
      params,
      dist,
      sess,
      return_per_pixel=return_per_pixel,
      dist_family=FLAGS.dist_family,
      wasserstein=True,
      condition_count=condition_count)
  grad_in = preds_in['grads']
  grad_ood = preds_ood['grads']
  grad_in = np.array(grad_in)
  grad_ood = np.array(grad_ood)
  print(grad_in.shape, grad_ood.shape)
  np.save(f'{FLAGS.model_dir}/grad_in', grad_in)
  np.save(f'{FLAGS.model_dir}/grad_ood', grad_ood)
  # save locs and scales
  locs_in = preds_in['locs']
  locs_ood = preds_ood['locs']
  locs_in = np.array(locs_in)
  locs_ood = np.array(locs_ood)
  print(locs_in.shape, locs_ood.shape)
  np.save(f'{FLAGS.model_dir}/locs_in', locs_in)
  np.save(f'{FLAGS.model_dir}/locs_ood', locs_ood)
  scales_in = preds_in['scales']
  scales_ood = preds_ood['scales']
  scales_in = np.array(scales_in)
  scales_ood = np.array(scales_ood)
  print(scales_in.shape, scales_ood.shape)
  np.save(f'{FLAGS.model_dir}/scales_in', scales_in)
  np.save(f'{FLAGS.model_dir}/scales_ood', scales_ood)
  log_probs_in = preds_in['log_probs']
  log_probs_ood = preds_ood['log_probs']
  np.save(f'{FLAGS.model_dir}/log_probs_in', log_probs_in)
  np.save(f'{FLAGS.model_dir}/log_probs_ood', log_probs_ood)
  emds_in = preds_in['emds']
  emds_ood = preds_ood['emds']
  np.save(f'{FLAGS.model_dir}/emds_in', emds_in)
  np.save(f'{FLAGS.model_dir}/emds_ood', emds_ood)
  images_in = preds_in['images']
  images_ood = preds_ood['images']
  np.save(f'{FLAGS.model_dir}/images_in', images_in)
  np.save(f'{FLAGS.model_dir}/images_ood', images_ood)
  if return_per_pixel:
    log_probs_per_pixel_in = preds_in['log_probs_per_pixel']
    log_probs_per_pixel_ood = preds_ood['log_probs_per_pixel']
    np.save(f'{FLAGS.model_dir}/log_probs_per_pixel_in', log_probs_per_pixel_in)
    np.save(f'{FLAGS.model_dir}/log_probs_per_pixel_ood', log_probs_per_pixel_ood)
    if 'emds_per_pixel' in preds_in:
      np.save(f'{FLAGS.model_dir}/emds_per_pixel_in', preds_in['emds_per_pixel'])
      np.save(f'{FLAGS.model_dir}/emds_per_pixel_ood', preds_ood['emds_per_pixel'])
  return preds_in, preds_ood, grad_in, grad_ood


def print_and_write(f, context):
  print(context + '\n')
  f.write(context + '\n')


def plot_heatmap(n, data, plt_file, colorbar=True):
  """Plot heatmaps (Figure 3 in the paper)."""
  sns.set_style('whitegrid')
  sns.set(style='ticks', rc={'lines.linewidth': 4})
  cmap_reversed = ListedColormap(sns.color_palette('Greys_r', 6).as_hex())
  fig, axes = plt.subplots(nrows=n, ncols=n, figsize=(2 * n - 2, 2 * n - 2))
  i = 0
  for ax in axes.flat:
    im = ax.imshow(data[i], vmin=0, vmax=6, cmap=cmap_reversed)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    i += 1
  fig.subplots_adjust(right=0.9)
  if colorbar:
    cbar_ax = fig.add_axes([0.95, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    cbar_ax.tick_params(labelsize=20)

  with tf.gfile.Open(plt_file, 'wb') as sp:
    plt.savefig(sp, format='pdf', bbox_inches='tight')

def main(unused_argv):
  # import pandas as pd
  # df = pd.read_csv(output_file)
  # if df.shape[0] > 0:
  #   df.columns=['model', 'exp', 'norm', 'auc']
  #   results = df[(df.model==FLAGS.model_dir)&(df.exp==FLAGS.exp)]
  #   if results.shape[0] > 0:
  #     print(f"{FLAGS.model_dir}, {FLAGS.exp} already run")

  # write results to file
  out_dir = os.path.join(FLAGS.model_dir, 'results')
  tf.compat.v1.gfile.MakeDirs(out_dir)
  out_f = tf.compat.v1.gfile.Open(
      os.path.join(out_dir, 'run%d.txt' % FLAGS.repeat_id), 'w')

  ## Final test on FashionMNIST-MNIST/CIFAR-SVHN
  # foreground model
  preds_in, preds_ood, grad_in, grad_ood = load_data_and_model_and_pred(
      FLAGS.exp,
      FLAGS.data_dir,
      0.0,
      0.0,
      FLAGS.repeat_id,
      FLAGS.ckpt_step,
      'test',
      'test',
      return_per_pixel=True)
  # np.save('in_img.npy', preds_in['images'][:5])
  # np.save('ood_img.npy', preds_ood['images'][:5])
  # print(preds_in['log_probs'][:5], preds_ood['log_probs'][:5])
  # auc = utils.compute_auc(
  #   preds_in['log_probs'], preds_ood['log_probs'], pos_label=0)
  # print(sum(preds_in['log_probs']))
  # print(auc)
  # import sys; sys.exit()
  
  print(sum(preds_in['log_probs']))
  auc = utils.compute_auc(
      preds_in['log_probs'], preds_ood['log_probs'], pos_label=0)
  print('mle auc: ', auc)
  # same thing
  # auc = utils.compute_auc(
  #     -preds_in['emds'], -preds_ood['emds'], pos_label=0)
  auc = utils.compute_auc(
      preds_in['emds'], preds_ood['emds'], pos_label=1)
  print('emd auc: ', auc)
  import sys; sys.exit()
  with open('evals_likelihood', 'a') as f:
    f.write('{}: {}\n'.format(FLAGS.model_dir, sum(preds_in['log_probs'])))

  def dist(preds):
    """ Doesn't care about whether the image is in the support """
    # (num_batches, batch_size, 28, 28, 1)
    preds['locs'] = np.concatenate(preds['locs'])
    preds['scales'] = np.concatenate(preds['scales'])
    preds['images'] = np.array(preds['images'])
    print('locs', preds['locs'].shape)
    print('scales', preds['scales'].shape)
    print('images', preds['images'].shape)
    # return np.minimum(np.abs(preds['images'] - preds['locs']), np.abs(preds['images'] - preds['scales'])).sum(axis=(1, 2, 3))
    return (np.abs(preds['images'] - preds['locs']) + np.abs(preds['images'] - preds['scales'])).sum(axis=(1, 2, 3))

  # auc_dist = utils.compute_auc(
  #     -dist(preds_in), -dist(preds_ood), pos_label=0)

  def emd(mins, maxes, label, norm, per_image):
    """ Note: currently only looking at as many images as exist in the in dist sample """
    mins = np.minimum(np.maximum(mins, np.zeros_like(mins)), np.ones_like(mins) * 255)
    maxes = np.minimum(np.maximum(maxes, np.zeros_like(maxes)), np.ones_like(maxes) * 255)
    # support distance, not wasserstein
    if norm == 0:
      loss = np.abs(label - (mins + maxes)/2)
    if norm == 1:
      loss_outside = np.abs(mins - label) + np.abs(maxes - label)
      loss_inside = [np.square(label) - np.multiply(label, (mins + maxes)) + np.divide(np.square(mins) + np.square(maxes), 2)]
      loss_inside = np.divide(loss_inside, np.abs(maxes - mins))
      loss = np.where(
        np.logical_and(
          np.less(label, np.maximum(mins, maxes)), np.greater(label, np.minimum(mins, maxes))),
        np.squeeze(loss_inside, 0),
        loss_outside
      )
    elif norm == 2:
      loss = (np.square(label)
      - label * (mins + maxes)
      + (np.square(mins) + np.square(maxes) + np.multiply(mins, maxes))/3
      )
      # print('loss', np.array(loss).shape, mins.shape, maxes.shape, label.shape)
    if per_image:
      return np.sum(loss, axis=(1, 2, 3))
    else:
      return np.sum(loss)
  # norm = int(FLAGS.model_dir.split('/expfashion')[0][-1])
  # print('norm:', norm)
  preds_in['locs'] = np.concatenate(preds_in['locs'])
  preds_in['scales'] = np.concatenate(preds_in['scales'])
  preds_ood['locs'] = np.concatenate(preds_ood['locs'])
  preds_ood['scales'] = np.concatenate(preds_ood['scales'])
  if FLAGS.dist_family == 'logistic':
    for norm in [2, 0]:
      emd_in = []
      emd_ood = []
      emd_in_bounded = []
      emd_ood_bounded = []
      for i in range(len(preds_in['locs'])):
        flattened_locs_in = preds_in['locs'][i].flatten()
        flattened_scales_in = preds_in['scales'][i].flatten()
        flattened_images_in = preds_in['images'][i].flatten()
        flattened_locs_ood = preds_ood['locs'][i].flatten()
        flattened_scales_ood = preds_ood['scales'][i].flatten()
        flattened_images_ood = preds_ood['images'][i].flatten()
        logistic_in_samples = np.random.logistic(flattened_locs_in, flattened_scales_in, size=(100, flattened_locs_in.shape[0]))
        logistic_ood_samples = np.random.logistic(flattened_locs_ood, flattened_scales_ood, size=(100, flattened_locs_ood.shape[0]))

        logistic_in_samples_bounded = logistic_in_samples.copy()
        logistic_ood_samples_bounded = logistic_ood_samples.copy()
        logistic_in_samples_bounded = np.maximum(logistic_in_samples_bounded, np.zeros_like(logistic_in_samples_bounded))
        logistic_in_samples_bounded = np.minimum(logistic_in_samples_bounded, np.ones_like(logistic_in_samples_bounded) * 255)
        logistic_ood_samples_bounded = np.maximum(logistic_ood_samples_bounded, np.zeros_like(logistic_ood_samples_bounded))
        logistic_ood_samples_bounded = np.minimum(logistic_ood_samples_bounded, np.ones_like(logistic_ood_samples_bounded) * 255)

        # keeps track of all pixel emds
        emd_p_in = []
        emd_p_ood = []
        emd_p_in_bounded = []
        emd_p_ood_bounded = []
        for j in range(flattened_locs_in.shape[0]):
          # print('num_samples ', logistic_in_samples[:, j].shape)
          if norm == 0:
            pixel_emd_in = np.abs(logistic_in_samples[:, j] - [flattened_images_in[j]]).mean()
            pixel_emd_ood = np.abs(logistic_ood_samples[:, j] - [flattened_images_ood[j]]).mean()
          if norm == 1:
            pixel_emd_in = ot.emd2_1d(logistic_in_samples[:, j], [flattened_images_in[j]], metric='minkowski')
            pixel_emd_ood = ot.emd2_1d(logistic_ood_samples[:, j], [flattened_images_ood[j]], metric='minkowski')
          elif norm == 2:
            pixel_emd_in = ot.emd2_1d(logistic_in_samples[:, j], [flattened_images_in[j]])
            pixel_emd_ood = ot.emd2_1d(logistic_ood_samples[:, j], [flattened_images_ood[j]])
            pixel_emd_in_bounded = ot.emd2_1d(logistic_in_samples_bounded[:, j], [flattened_images_in[j]])
            pixel_emd_ood_bounded = ot.emd2_1d(logistic_ood_samples_bounded[:, j], [flattened_images_ood[j]])
          emd_p_in.append(pixel_emd_in)
          emd_p_ood.append(pixel_emd_ood)
          emd_p_in_bounded.append(pixel_emd_in_bounded)
          emd_p_ood_bounded.append(pixel_emd_ood_bounded)
        emd_in.append(emd_p_in)
        emd_ood.append(emd_p_ood)
        emd_in_bounded.append(emd_p_in_bounded)
        emd_ood_bounded.append(emd_p_ood_bounded)
      print('emd_in', np.array(emd_in).shape)
      print('emd_ood', np.array(emd_ood).shape)
      np.save(f'{FLAGS.model_dir}/emd_cont_in_{norm}', emd_in)
      np.save(f'{FLAGS.model_dir}/emd_cont_ood_{norm}', emd_ood)
      np.save(f'{FLAGS.model_dir}/emd_cont_in_bounded_{norm}', emd_in_bounded)
      np.save(f'{FLAGS.model_dir}/emd_cont_ood_bounded_{norm}', emd_ood_bounded)
      auc_dist = utils.compute_auc(
        -np.array(emd_in).sum(axis=1),
        -np.array(emd_ood).sum(axis=1),
        pos_label=0
      )
      print('w norm {} {}'.format(norm, auc_dist))
      auc_dist_bounded = utils.compute_auc(
        -np.array(emd_in_bounded).sum(axis=1),
        -np.array(emd_ood_bounded).sum(axis=1),
        pos_label=0
      )
      print('w norm bounded {} {}'.format(norm, auc_dist_bounded))
      with open(output_file, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([FLAGS.model_dir, FLAGS.exp, norm, auc_dist])
      #   # f.write('{} w{}: {}\n'.format(FLAGS.model_dir, norm, auc_dist))
  elif FLAGS.dist_family == 'uniform':
    for norm in [0, 1, 2]:
      emd_in = []
      emd_ood = []
      for i in range(0, len(preds_in['locs']), 50):
        emd_in.extend(emd(preds_in['locs'][i:i+50], preds_in['scales'][i:i+50], preds_in['images'][i:i+50], norm=norm, per_image=True))
        emd_ood.extend(emd(preds_ood['locs'][i:i+50], preds_ood['scales'][i:i+50], preds_ood['images'][i:i+50], norm=norm, per_image=True))
      print('emd_in', np.array(emd_in).min(), np.array(emd_in).max())
      print('emd_ood', np.array(emd_ood).min(), np.array(emd_ood).max())
      print('preds_in', preds_in['locs'].min(),preds_in['locs'].max(), preds_in['scales'].min(), preds_in['scales'].max())
      print('preds_ood', preds_ood['locs'].min(),preds_ood['locs'].max(), preds_ood['scales'].min(), preds_ood['scales'].max())
      print(np.array(emd_in).shape, np.array(emd_ood).shape)
      auc_dist = utils.compute_auc(
        -np.array(emd_in),
        -np.array(emd_ood),
        pos_label=0
      )
      print('w norm {} {}'.format(norm, auc_dist))
      with open(output_file, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([FLAGS.model_dir, FLAGS.exp, norm, auc_dist])
      # with open(output_file, 'a') as f:
      #   f.write('{} w{}: {}\n'.format(FLAGS.model_dir, norm, auc_dist))
      # with open('evals2.txt', 'a') as f:
      #   f.write('{}: {}\n'.format(FLAGS.model_dir, auc_dist))
  auc = utils.compute_auc(
      preds_in['log_probs'], preds_ood['log_probs'], pos_label=0)
  with open(output_file, 'a') as f:
    writer = csv.writer(f)
    writer.writerow([FLAGS.model_dir, FLAGS.exp, 'mle', auc])
  # with open(output_file, 'a') as f:
  #     f.write('{} mle: {}\n'.format(FLAGS.model_dir, auc))
  # with open('evals_mle2.txt', 'a') as f:
  #   f.write('{}: {}\n'.format(FLAGS.model_dir, auc))
  print('mle {}'.format(auc))

if __name__ == '__main__':
  app.run(main)
