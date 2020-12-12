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

from genomics_ood.images_ood import pixel_cnn
from genomics_ood.images_ood import utils

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
  if repeat_id == -1:
    ckpt_repeat_dir = os.path.join(ckpt_dir, 'model')
  else:
    # each param_dir may have multiple independent runs
    try:
      repeat_dir_list = tf.compat.v1.gfile.ListDirectory(ckpt_dir)
    except tf.errors.NotFoundError:
      return None
    repeat_dir = repeat_dir_list[repeat_id]
    ckpt_repeat_dir = os.path.join(ckpt_dir, repeat_dir, 'model')
  ckpt_file = utils.get_ckpt_at_step(ckpt_repeat_dir, ckpt_step)
  # print('ckpt_file={}'.format(ckpt_file))
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
                                 eval_mode,
                                 return_per_pixel=False):
  """Load datasets, load model ckpt, and eval the model on the datasets."""
  tf.compat.v1.reset_default_graph()
  # load datasets
  datasets = load_datasets(exp, data_dir)
  # load model
  ckpt_file = find_ckpt_match_param(reg_weight, mutation_rate, repeat_id,
                                    ckpt_step)
  if not ckpt_file:  # no ckpt file is found
    # raise ValueError('No ckpt model found')
    return None, None, None, None

  dist, params, sess = create_model_and_restore_ckpt(ckpt_file)

  # Evaluations
  preds_in = utils.eval_on_data(
      datasets['%s_in' % eval_mode],
      utils.image_preprocess,
      params,
      dist,
      sess,
      return_per_pixel=return_per_pixel)
  if eval_mode == 'val':
    if exp in ['fashion', 'mnist']:
      data = datasets['val_ood']
    else:
      data = datasets['val_in']
  elif eval_mode == 'test':
    data = datasets['test_ood']
  else:
    raise ValueError("Bad eval_mode: ", eval_mode)
  preds_ood = utils.eval_on_data(
      data,
      utils.image_preprocess,
      params,
      dist,
      sess,
      return_per_pixel=return_per_pixel)
  # grad_in = tape.gradient(datasets['%s_in' % eval_mode], preds_in['log_probs'])
  # grad_ood = tape.gradient(data, preds_ood['log_probs'])
  grad_in = preds_in.pop('grads')
  grad_ood = preds_ood.pop('grads')
  grad_in = np.array(grad_in)
  grad_ood = np.array(grad_ood)
  print(grad_in.shape, grad_ood.shape)
  np.save(f'grad_in_{exp}', grad_in)
  np.save(f'grad_ood_{exp}', grad_ood)
  # grad_in = tf.norm(grad_in.reshape((grad_in.shape[0], -1)), axis=1)
  # grad_ood = tf.norm(grad_ood.reshape((grad_ood.shape[0], -1)), axis=1)
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

def calculate_zeros(exp, data_dir):
  if exp == 'fashion':
    test_in = os.path.join(data_dir, 'fashion_mnist_test.npy')
    test_ood = os.path.join(data_dir, 'mnist_test.npy')
  elif exp == 'mnist':
    test_in = os.path.join(data_dir, 'mnist_test.npy')
    test_ood = os.path.join(data_dir, 'fashion_mnist_test.npy')
  else:
    raise ValueError("exp not supported: ", exp)
  img_in = np.load(test_in)
  img_ood = np.load(test_ood)
  img_in = img_in.reshape((img_in.shape[0], -1))
  img_ood = img_ood.reshape((img_ood.shape[0], -1))
  zeros_in = (img_in == 0).sum(axis=1) / img_in.shape[1]
  zeros_ood = (img_ood == 0).sum(axis=1) / img_ood.shape[1]
  return zeros_in, zeros_ood

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
      return_per_pixel=True)

  auc = utils.compute_auc(
      preds_in['log_probs'], preds_ood['log_probs'], pos_label=0)
  if FLAGS.exp in ['fashion', 'mnist']:
    zeros_in, zeros_ood = calculate_zeros(FLAGS.exp, FLAGS.data_dir)
  else:
    zeros_in, zeros_ood = calculate_complexity(FLAGS.exp, FLAGS.data_dir)
  plt.scatter(zeros_in, preds_in['log_probs'], color='blue', alpha=.2)
  plt.scatter(zeros_ood, preds_ood['log_probs'], color='red', alpha=.2)
  plt.title(FLAGS.exp + ' likelihood')
  plt.savefig(FLAGS.exp + ' likelihood' + '.pdf', bbox_inches='tight')
  plt.clf()
  print_and_write(out_f, 'final test, auc={}'.format(auc))

  # typicality approximation
  grad_in = grad_in.reshape((-1))
  grad_ood = grad_ood.reshape((-1))
  grad_auc = utils.compute_auc(
      grad_in, grad_ood, pos_label=0)
  print(zeros_in.shape, grad_in.shape)
  plt.scatter(zeros_in, grad_in, color='blue', alpha=.2)
  plt.scatter(zeros_ood, grad_ood, color='red', alpha=.2)
  plt.title(FLAGS.exp + ' typicality')
  plt.savefig(FLAGS.exp + ' typicality' + '.pdf', bbox_inches='tight')
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
