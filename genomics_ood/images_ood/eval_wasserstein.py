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

from genomics_ood.images_ood import pixel_cnn
from genomics_ood.images_ood import utils

from scipy.stats import wasserstein_distance
import ot

tf.compat.v1.disable_v2_behavior()

flags.DEFINE_string('model_dir', '/tmp/expfashion/rescaleFalse/',
                    'Directory to write results and logs.')
flags.DEFINE_string('data_dir', '/tmp/image_data',
                    'Directory to data np arrays.')
flags.DEFINE_integer('ckpt_step', 10, 'The step of the selected ckpt.')
flags.DEFINE_string('exp', 'fashion', 'cifar or fashion')
flags.DEFINE_integer(
    'repeat_id', -1,
    ('We run 10 independent experiments to get the mean and variance of AUROC.',
     'repeat_id=i indicates the i-th independent run.',
     'repeat_id=-1 indecates only one independent run.'))
flags.DEFINE_boolean('color_classes', False, 'whether to color by class (only for fashion mnist and mnist)')
flags.DEFINE_boolean('logistic', False, 'whether model is logistic')
FLAGS = flags.FLAGS

REG_WEIGHT_LIST = [0, 10, 100]
MUTATION_RATE_LIST = [0.1, 0.2, 0.3]


def load_datasets(exp, data_dir):
  if exp == 'fashion':
    datasets = utils.load_fmnist_datasets(data_dir)
  elif exp == 'mnist':
    datasets = utils.load_mnist_datasets(data_dir)
  elif exp == 'svhn':
    datasets = utils.load_svhn_datasets(data_dir)
  else:
    datasets = utils.load_cifar_datasets(data_dir)
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
    # raise ValueError('No ckpt model found')
    return None, None, None, None

  dist, params, sess = create_model_and_restore_ckpt(ckpt_file)

  # Evaluations
  preds_in = utils.eval_on_data(
      datasets['%s_in' % eval_mode_in],
      utils.image_preprocess,
      params,
      dist,
      sess,
      return_per_pixel=return_per_pixel,
      wasserstein=True if not FLAGS.logistic else False)
  preds_ood = utils.eval_on_data(
      datasets['%s_ood' % eval_mode_ood],
      utils.image_preprocess,
      params,
      dist,
      sess,
      return_per_pixel=return_per_pixel,
      wasserstein=True if not FLAGS.logistic else False)
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
  return preds_in, preds_ood, grad_in, grad_ood


def compute_auc_llr(preds_in, preds_ood, preds0_in, preds0_ood):
  """Compute AUC for LLR."""
  # check if samples are in the same order
  assert np.array_equal(preds_in['labels'], preds0_in['labels'])
  assert np.array_equal(preds_ood['labels'], preds0_ood['labels'])

  # evaluate AUROC for OOD detection
  auc = utils.compute_auc(
      preds_in['log_probs'], preds_ood['log_probs'], pos_label=0)
  llr_in = preds_in['log_probs'] - preds0_in['log_probs']
  llr_ood = preds_ood['log_probs'] - preds0_ood['log_probs']
  auc_llr = utils.compute_auc(llr_in, llr_ood, pos_label=0)
  return auc, auc_llr

def compute_auc_grad(preds_in, preds_ood, preds0_in, preds0_ood):
  """Compute AUC for LLR."""

  # evaluate AUROC for OOD detection
  auc = utils.compute_auc(
      preds_in, preds_ood, pos_label=0)
  llr_in = preds_in - preds0_in
  llr_ood = preds_ood - preds0_ood
  auc_llr = utils.compute_auc(llr_in, llr_ood, pos_label=0)
  return auc, auc_llr

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

def calculate_zeros(exp, data_dir, eval_mode_in='test', eval_mode_ood='test'):
  eval_mode_in = 'train' if eval_mode_in == 'tr' else eval_mode_in
  eval_mode_ood = 'train' if eval_mode_ood == 'tr' else eval_mode_ood
  if exp == 'fashion':
    test_in = os.path.join(data_dir, f'fashion_mnist_{eval_mode_in}.npy')
    test_ood = os.path.join(data_dir, f'mnist_{eval_mode_ood}.npy')
    print(eval_mode_ood)
  elif exp == 'mnist':
    test_in = os.path.join(data_dir, f'mnist_{eval_mode_in}.npy')
    test_ood = os.path.join(data_dir, f'fashion_mnist_{eval_mode_ood}.npy')
  else:
    raise ValueError("exp not supported: ", exp)
  img_in = np.load(test_in)
  img_ood = np.load(test_ood)
  print(img_ood.shape)
  img_in = img_in.reshape((img_in.shape[0], -1))
  img_ood = img_ood.reshape((img_ood.shape[0], -1))
  zeros_in = (img_in == 0).sum(axis=1) / img_in.shape[1]
  zeros_ood = (img_ood == 0).sum(axis=1) / img_ood.shape[1]
  # zeros_in = np.mean(img_in, axis=1)
  # zeros_ood = np.mean(img_ood, axis=1)
  return zeros_in, zeros_ood

def get_classes(exp, data_dir):
  if exp == 'fashion':
    in_name = 'fashion_mnist'
    ood_name = 'mnist'
  elif exp == 'mnist':
    in_name = 'mnist'
    ood_name = 'fashion_mnist'
  else:
    raise ValueError('bad exp')
  in_classes = np.load(os.path.join(data_dir, in_name + '_labels1.npy'))
  ood_classes = np.load(os.path.join(data_dir, ood_name + '_labels1.npy'))
  return in_classes, ood_classes

def calculate_complexity(exp, data_dir):
  if exp == 'fashion':
    test_in = os.path.join(data_dir, 'fashion_mnist_test.npy')
    test_ood = os.path.join(data_dir, 'mnist_test.npy')
  elif exp == 'mnist':
    test_in = os.path.join(data_dir, 'mnist_test.npy')
    test_ood = os.path.join(data_dir, 'fashion_mnist_test.npy')
  elif exp == 'svhn':
    test_in = os.path.join(data_dir, 'cifar10_test.npy')
    test_ood = os.path.join(data_dir, 'svhn_cropped_test.npy')
  elif exp == 'cifar':
    test_in = os.path.join(data_dir, 'svhn_cropped_test.npy')
    test_ood = os.path.join(data_dir, 'cifar10_test.npy')
  else:
    raise ValueError("exp not supported: ", exp)
  

def main(unused_argv):


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
  
  print(sum(preds_in['log_probs']))
  with open('evals_likelihood', 'a') as f:
    f.write('{}: {}\n'.format(FLAGS.model_dir, sum(preds_in['log_probs'])))
  import sys; sys.exit()

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
      loss = [np.square(label)
      - label * (mins + maxes)
      + (np.square(mins) + np.square(maxes) + np.multiply(mins, maxes))/3
      ]
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
  if FLAGS.logistic:
    for norm in [0, 1, 2]:
      emd_in = []
      emd_ood = []
      for i in range(len(preds_in['locs'])):
        flattened_locs_in = preds_in['locs'][i].flatten()
        flattened_scales_in = preds_in['scales'][i].flatten()
        flattened_images_in = preds_in['images'][i].flatten()
        flattened_locs_ood = preds_ood['locs'][i].flatten()
        flattened_scales_ood = preds_ood['scales'][i].flatten()
        flattened_images_ood = preds_ood['images'][i].flatten()
        logistic_in_samples = np.random.logistic(flattened_locs_in, flattened_scales_in, size=(100, flattened_locs_in.shape[0]))
        logistic_ood_samples = np.random.logistic(flattened_locs_ood, flattened_scales_ood, size=(100, flattened_locs_ood.shape[0]))
        pixel_emd_in = 0
        pixel_emd_ood = 0
        for j in range(flattened_locs_in.shape[0]):
          # print('num_samples ', logistic_in_samples[:, j].shape)
          if norm == 0:
            pixel_emd_in += np.abs(logistic_in_samples[:, j] - [flattened_images_in[j]]).mean()
            pixel_emd_ood += np.abs(logistic_ood_samples[:, j] - [flattened_images_ood[j]]).mean()
          if norm == 1:
            pixel_emd_in += ot.emd2_1d(logistic_in_samples[:, j], [flattened_images_in[j]], metric='minkowski')
            pixel_emd_ood += ot.emd2_1d(logistic_ood_samples[:, j], [flattened_images_ood[j]], metric='minkowski')
          elif norm == 2:
            pixel_emd_in += ot.emd2_1d(logistic_in_samples[:, j], [flattened_images_in[j]])
            pixel_emd_ood += ot.emd2_1d(logistic_ood_samples[:, j], [flattened_images_ood[j]])
        emd_in.append(pixel_emd_in)
        emd_ood.append(pixel_emd_ood)
      print('emd_in', np.array(emd_in).shape)
      print('emd_ood', np.array(emd_ood).shape)
      auc_dist = utils.compute_auc(
        -np.array(emd_in),
        -np.array(emd_ood),
        pos_label=0
      )
      print('w norm {} {}'.format(norm, auc_dist))
      with open('evals_table6', 'a') as f:
        f.write('{} w{}: {}\n'.format(FLAGS.model_dir, norm, auc_dist))
  else:
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
      auc_dist = utils.compute_auc(
        -np.array(emd_in),
        -np.array(emd_ood),
        pos_label=0
      )
      print('w norm {} {}'.format(norm, auc_dist))
      with open('evals_table6', 'a') as f:
        f.write('{} w{}: {}\n'.format(FLAGS.model_dir, norm, auc_dist))
      # with open('evals2.txt', 'a') as f:
      #   f.write('{}: {}\n'.format(FLAGS.model_dir, auc_dist))
  auc = utils.compute_auc(
      preds_in['log_probs'], preds_ood['log_probs'], pos_label=0)
  with open('evals_table6', 'a') as f:
      f.write('{} mle: {}\n'.format(FLAGS.model_dir, auc))
  # with open('evals_mle2.txt', 'a') as f:
  #   f.write('{}: {}\n'.format(FLAGS.model_dir, auc))
  print('mle {}'.format(auc))

  import sys; sys.exit()


  if FLAGS.exp in ['fashion', 'mnist']:
    zeros_in, zeros_ood = calculate_zeros(FLAGS.exp, FLAGS.data_dir, 'test', 'test1')
  else:
    zeros_in, zeros_ood = calculate_complexity(FLAGS.exp, FLAGS.data_dir)
  print(len(zeros_ood), len(preds_ood['log_probs']))
  plt.scatter(zeros_in, preds_in['log_probs'], color='blue', alpha=.2)
  plt.scatter(zeros_ood, preds_ood['log_probs'], color='red', alpha=.2)
  plt.title(FLAGS.exp + ' likelihood')
  plt.savefig(os.path.join(out_dir, FLAGS.exp + ' likelihood' + '.pdf'), bbox_inches='tight')
  plt.clf()
  # print_and_write(out_f, 'final test, auc={}'.format(auc))
  if FLAGS.color_classes:
    def to_labels(in_classes, ood_classes, exp):
        fashion_dict = {0: 'tshirt', 1: 'trouser', 2: 'pullover', 3:'dress', 4: 'coat', 5:'sandal', 6:'shirt', 7:'sneaker', 8:'bag', 9:'ankle_boot'}
        mnist_dict = {k: str(k) for k in range(10)}
        if exp == 'fashion':
            in_dict = fashion_dict
            ood_dict = mnist_dict
        elif exp == 'mnist':
            in_dict = mnist_dict
            ood_dict = fashion_dict
        else:
            raise ValueError('bad exp')
        # in_labels = [in_dict[i] for i in in_classes]
        # ood_labels = [ood_dict[i] for i in ood_classes]
        return in_labels, ood_labels
    in_classes, ood_classes = get_classes(FLAGS.exp, FLAGS.data_dir)
    plt.scatter(zeros_in, preds_in['log_probs'], c=in_classes, cmap='Pastel2', alpha=.2)
    plt.scatter(zeros_ood, preds_ood['log_probs'], c=ood_classes, cmap='Dark2', alpha=.2)
  else:
    plt.scatter(zeros_in, preds_in['log_probs'], color='blue', alpha=.2)
    plt.scatter(zeros_ood, preds_ood['log_probs'], color='red', alpha=.2)
  plt.title(FLAGS.exp + ' likelihood')
  lines_in = [Line2D([0], [0], color=plt.cm.Pastel2(i)) for i in range(10)]
  lines_ood = [Line2D([0], [0], color=plt.cm.Dark2(i)) for i in range(10)]
  lines = lines_in + lines_ood
  fashion_labels = ['tshirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle_boot']
  mnist_labels = list(range(10))
  if FLAGS.exp == 'fashion':
    labels = fashion_labels + mnist_labels
  elif FLAGS.exp == 'mnist':
    labels = mnist_labels + fashion_labels
  plt.legend(lines, labels)
  plt.savefig(os.path.join(out_dir, FLAGS.exp + ' likelihood colored.pdf'), bbox_inches='tight')
  plt.clf()

  # typicality approximation
  grad_in = grad_in.reshape((-1))
  grad_ood = grad_ood.reshape((-1))
  grad_auc = utils.compute_auc(
      grad_in, grad_ood, pos_label=0)
  print(zeros_in.shape, grad_in.shape)
  plt.scatter(zeros_in, grad_in, color='blue', alpha=.2)
  plt.scatter(zeros_ood, grad_ood, color='red', alpha=.2)
  plt.title(FLAGS.exp + ' typicality')
  plt.savefig(os.path.join(out_dir, FLAGS.exp + ' typicality' + '.pdf'), bbox_inches='tight')
  plt.clf()
  print_and_write(out_f, 'final test grad, auc={}'.format(grad_auc))

  out_f.close()

  # plot heatmaps (Figure 3)
  if FLAGS.exp in ['fashion', 'mnist']:
    n = 4

    # FashionMNIST
    log_probs_in = preds_in['log_probs']
    log_probs_pp_in = preds_in['log_probs_per_pixel']
    n_sample_in = len(log_probs_in)
    log_probs_in_sorted = sorted(
        range(n_sample_in), key=lambda k: log_probs_in[k], reverse=True)
    ids_seq = np.arange(1, n_sample_in, int(n_sample_in / (n * n)))

    ## pure likelihood
    data = [
        log_probs_pp_in[log_probs_in_sorted[ids_seq[i]]] + 6
        for i in range(n * n)
    ]
    plt_file = os.path.join(
        out_dir, f'run%d_heatmap_{FLAGS.exp}_test_in_p(x).pdf' % FLAGS.repeat_id)
    plot_heatmap(n, data, plt_file)

    # MNIST
    log_probs_ood = preds_ood['log_probs']
    log_probs_pp_ood = preds_ood['log_probs_per_pixel']
    n_sample_ood = len(log_probs_ood)
    log_probs_ood_sorted = sorted(
        range(n_sample_ood), key=lambda k: log_probs_ood[k], reverse=True)
    ids_seq = np.arange(1, n_sample_ood, int(n_sample_ood / (n * n)))

    ## pure likelihood
    data = [
        log_probs_pp_ood[log_probs_ood_sorted[ids_seq[i]]] + 6
        for i in range(n * n)
    ]
    plt_file = os.path.join(out_dir,
                            f'run%d_heatmap_{FLAGS.exp}_test_ood_p(x).pdf' % FLAGS.repeat_id)
    plot_heatmap(n, data, plt_file)


if __name__ == '__main__':
  app.run(main)
