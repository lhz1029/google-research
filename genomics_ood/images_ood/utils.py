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

# Lint as: python3
"""Utils for training pixel_cnn model for images."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
from sklearn import metrics
import tensorflow.compat.v1 as tf
import yaml
from PIL import Image
import glob
import pickle
import cv2


def load_tfdata_from_np(np_file, flip=None, binarize=False):
  # try:
  #   with tf.compat.v1.gfile.Open(np_file, mode='rb') as f:
  #     images = np.load(f)
  #     labels = np.load(f)
  # except ValueError:
  #   print('tf open failed')
  images = np.load(np_file)
  if flip == 'v':
    images = np.array([np.flipud(img) for img in images])
  elif flip == 'h':
    images = np.array([np.fliplr(img) for img in images])
  if binarize:
    print('binarizing')
    print('before sum: ', images.sum())
    # bin2
    images[:] = np.where(images > 127.5, 255, 0)
    # binarize flip
    # images[:] = np.where(images > 127.5, 0, 255)
    # binarize 0, 1; bin1
    # images[:] = np.where(images > 127.5, 1, 0)
    print('sum: ', images.sum())
    assert len(np.unique(images.flatten())) == 2, f"Too many pixel values: {np.unique(images.flatten())}"
  labels = images
  dataset = tf.compat.v1.data.Dataset.from_tensor_slices(
      (images, labels)).map(tensor_slices_preprocess)
  return dataset

def load_single_pixel_datasets(data_dir=None):
  d = {}
  for key in ['tr_in', 'val_in']:
    images = np.random.logistic(loc=100, scale=10, size=(10000, 1, 1, 1))
    labels = images
    d[key] = tf.compat.v1.data.Dataset.from_tensor_slices(
    (images, labels)).map(tensor_slices_preprocess)
  return d

def load_ones(data_dir=None):
  d = {}
  for key in ['tr_in', 'val_in']:
    images = np.ones((128, 28, 28, 1))
    labels = images
    d[key] = tf.compat.v1.data.Dataset.from_tensor_slices(
    (images, labels)).map(tensor_slices_preprocess)
  return d

def load_fmnist_datasets(data_dir, out_data='mnist', binarize=False):
  """Load fashionMNIST and MNIST dataset from np array."""
  tr_in = load_tfdata_from_np(os.path.join(data_dir, 'fashion_mnist_train.npy'), binarize=binarize)
  val_in = load_tfdata_from_np(os.path.join(data_dir, 'fashion_mnist_val.npy'), binarize=binarize)
  test_in = load_tfdata_from_np(
      os.path.join(data_dir, 'fashion_mnist_test.npy'), binarize=binarize)
  test1_in = load_tfdata_from_np(os.path.join(data_dir, 'fashion_mnist_test1.npy'), binarize=binarize)

  if out_data == 'mnist':
    tr_ood = load_tfdata_from_np(os.path.join(data_dir, 'mnist_train.npy'), binarize=binarize)
    val_ood = load_tfdata_from_np(os.path.join(data_dir, 'notmnist.npy'), binarize=binarize)
    test1_ood = load_tfdata_from_np(os.path.join(data_dir, 'mnist_test1.npy'), binarize=binarize)
    test_ood = load_tfdata_from_np(os.path.join(data_dir, 'mnist_test.npy'), binarize=binarize)
  elif out_data == 'hflip':
    test_ood = load_tfdata_from_np(
      os.path.join(data_dir, 'fashion_mnist_test.npy'), flip='h', binarize=binarize)
  elif out_data == 'vflip':
    test_ood = load_tfdata_from_np(
      os.path.join(data_dir, 'fashion_mnist_test.npy'), flip='v', binarize=binarize)
  elif out_data == 'unif':
    images = np.stack([np.ones((28, 28, 1)) * i for i in range(256)], axis=0)
    labels = images
    test_ood = tf.compat.v1.data.Dataset.from_tensor_slices(
      (images, labels)).map(tensor_slices_preprocess)
  elif out_data == 'gaussian':
    images = np.random.normal(size=(10000, 28, 28, 3))
    labels = images
    test_ood = tf.compat.v1.data.Dataset.from_tensor_slices(
      (images, labels)).map(tensor_slices_preprocess)
  elif out_data == 'omniglot':
    images = []
    for img_file in glob.glob(os.path.join(data_dir, 'omniglot', 'images_evaluation') + '/**/*.png', recursive=True):
      img = cv2.imread(img_file)
      img = cv2.resize(img, (28, 28,)).astype('uint8')
      images.append(img)
    images = np.array(images)
    print(images.shape)
    images = images.reshape((-1, 28, 28, 1))
    labels = images
    test_ood = tf.compat.v1.data.Dataset.from_tensor_slices(
      (images, labels)).map(tensor_slices_preprocess)
  results = {
      'tr_in': tr_in,
      'val_in': val_in,
      'test_in': test_in,
      'test1_in': test1_in,
      'val_ood': val_ood,
      'test_ood': test_ood,
      'test1_ood': test1_ood
  }
  if out_data == 'mnist':
    results['tr_ood'] = tr_ood
  return results

def load_mnist_datasets(data_dir):
  """Load MNIST and fashionMNIST dataset from np array."""
  tr_in = load_tfdata_from_np(os.path.join(data_dir, 'mnist_train.npy'))
  val_in = load_tfdata_from_np(os.path.join(data_dir, 'mnist_val.npy'))
  test_in = load_tfdata_from_np(
      os.path.join(data_dir, 'mnist_test.npy'))
  test1_in = load_tfdata_from_np(os.path.join(data_dir, 'mnist_test1.npy'))

  val_ood = load_tfdata_from_np(os.path.join(data_dir, 'notmnist.npy'))
  test_ood = load_tfdata_from_np(os.path.join(data_dir, 'fashion_mnist_test.npy'))
  test1_ood = load_tfdata_from_np(os.path.join(data_dir, 'fashion_mnist_test1.npy'))
  return {
      'tr_in': tr_in,
      'val_in': val_in,
      'test_in': test_in,
      'test1_in': test1_in,
      'val_ood': val_ood,
      'test_ood': test_ood,
      'test1_ood': test1_ood
  }


def load_cifar_datasets(data_dir, out_data='svhn'):
  """Load CIFAR10 and SVHN dataset from np array."""
  tr_in = load_tfdata_from_np(os.path.join(data_dir, 'cifar10_train.npy'))
  val_in = load_tfdata_from_np(os.path.join(data_dir, 'cifar10_val.npy'))
  test_in = load_tfdata_from_np(os.path.join(data_dir, 'cifar10_test.npy'))

  if out_data == 'svhn':
    tr_ood = load_tfdata_from_np(os.path.join(data_dir, 'svhn_cropped_tr.npy'))
    test_ood = load_tfdata_from_np(os.path.join(data_dir, 'svhn_cropped_test.npy'))
  elif out_data == 'celeba':
    test_png_idx = range(182638, 202599 + 1)
    images = []
    for i in test_png_idx:
      img_file = os.path.join(data_dir, 'img_align_celeba', f'{i}.jpg')
      img = cv2.imread(img_file)
      img = cv2.resize(img, (32, 32,)).astype('uint8')
      images.append(np.asarray(img))
    images = np.array(images)
    labels = images
    test_ood = tf.compat.v1.data.Dataset.from_tensor_slices(
      (images, labels)).map(tensor_slices_preprocess)
  elif out_data == 'hflip':
    test_ood = load_tfdata_from_np(
      os.path.join(data_dir, 'cifar10_test.npy'), flip='h')
  elif out_data == 'vflip':
    test_ood = load_tfdata_from_np(
      os.path.join(data_dir, 'cifar10_test.npy'), flip='v')
  elif out_data == 'unif':
    images = np.random.uniform(size=(10000, 32, 32, 3))
    labels = images
    test_ood = tf.compat.v1.data.Dataset.from_tensor_slices(
      (images, labels)).map(tensor_slices_preprocess)
  elif out_data == 'gaussian':
    images = np.random.normal(size=(10000, 32, 32, 3))
    labels = images
    test_ood = tf.compat.v1.data.Dataset.from_tensor_slices(
      (images, labels)).map(tensor_slices_preprocess)
  elif out_data == 'cifar100':
    with open(os.path.join(data_dir, 'cifar-100-python', 'test'), 'rb') as f:
      data = pickle.load(f, encoding='bytes')[b'data']
      images = data.reshape((10000, 3, 32, 32))
      images = np.transpose(images, (0, 2, 3, 1))
      labels = images
    test_ood = tf.compat.v1.data.Dataset.from_tensor_slices(
    (images, labels)).map(tensor_slices_preprocess)
  elif out_data == 'imagenet32':
    raise NotImplemented("imagenet32 not yet set up")
  results = {
      'tr_in': tr_in,
      'val_in': val_in,
      'test_in': test_in,
      'test_ood': test_ood  # val_ood is val_in_grey
  }
  if out_data == 'svhn':
    results['tr_ood'] = tr_ood
  return results

def tensor_slices_preprocess(x, y):
  new = {}
  new['image'] = tf.cast(x, tf.float32)
  new['label'] = tf.cast(y, tf.int32)
  return new


def image_preprocess(x):
  x['image'] = tf.cast(x['image'], tf.float32)
  return x


def mutate_x(x, mutation_rate):
  """Add mutations to input.

  Generate mutations for all positions,
  in order to be different than itselves, the mutations have to be >= 1
  mute the untargeted positions by multiple mask (1 for targeted)
  then add the mutations to the original, mod 255 if necessary.

  Args:
    x: input image tensor of size batch*width*height*channel
    mutation_rate: mutation rate

  Returns:
    mutated input
  """
  w, h, c = x.get_shape().as_list()
  mask = tf.cast(
      tf.compat.v1.multinomial(
          tf.compat.v1.log([[1.0 - mutation_rate, mutation_rate]]), w * h * c),
      tf.int32)[0]
  mask = tf.reshape(mask, [w, h, c])
  possible_mutations = tf.compat.v1.random_uniform(
      [w * h * c],
      minval=0,
      maxval=256,  # 256 values [0, 1, ..., 256) = [0, 1, ..., 255]
      dtype=tf.int32)
  possible_mutations = tf.reshape(possible_mutations, [w, h, c])
  x = tf.compat.v1.mod(tf.cast(x, tf.int32) + mask * possible_mutations, 256)
  x = tf.cast(x, tf.float32)
  return x


def image_preprocess_add_noise(x, mutation_rate):
  """Image preprocess and add noise to image."""
  x['image'] = tf.cast(x['image'], tf.float32)

  if mutation_rate > 0:
    x['image'] = mutate_x(x['image'], mutation_rate)

  return x  # (input, output) of the model


def image_preprocess_grey(x):  # used for generate CIFAR-grey
  x['image'] = tf.compat.v1.image.rgb_to_grayscale(x['image'])
  x['image'] = tf.tile(x['image'], [1, 1, 3])
  x['image'] = tf.cast(x['image'], tf.float32)
  return x


def compute_auc(neg, pos, pos_label=1):
  ys = np.concatenate((np.zeros(len(neg)), np.ones(len(pos))), axis=0)
  neg = np.nan_to_num(neg)
  pos = np.nan_to_num(pos)
  neg = np.array(neg)[np.logical_not(np.isnan(neg))]
  pos = np.array(pos)[np.logical_not(np.isnan(pos))]
  scores = np.concatenate((neg, pos), axis=0)
  auc = metrics.roc_auc_score(ys, scores)
  if pos_label == 1:
    return auc
  else:
    return 1 - auc


def get_ckpt_at_step(tr_model_dir, step):
  import glob
  if step == -1:
    fnames = glob.glob(os.path.join(tr_model_dir, 'model_step*.ckpt.index'))
    steps = [int(f.split('step')[1].split('.')[0]) for f in fnames]
    step = max(steps)
    print('step', step)
  pattern = 'model_step{}.ckpt.index'.format(step)
  list_of_ckpt = tf.compat.v1.gfile.Glob(os.path.join(tr_model_dir, pattern))
  if list_of_ckpt:
    ckpt_file = list_of_ckpt[0].replace('.index', '')
    return ckpt_file
  else:
    tf.compat.v1.logging.fatal('Cannot find the ckpt file at step %s in dir %s',
                               step, tr_model_dir)
    return None


def load_hparams(params_yaml_file):
  """Create tf.HParams object based on params loaded from yaml file."""
  with tf.compat.v1.gfile.Open(params_yaml_file, mode='rb') as f:
    params = yaml.safe_load(f)
    params['dropout_rate'] = 0.0  # turn off dropout for eval

  return params


def get_count_dict():
  fashion = np.load('../data/fashion_mnist_train.npy')
  fashion = fashion.reshape((fashion.shape[0], -1))
  count_blacks = (fashion == 0).sum(axis=1)
  from collections import Counter
  count_dict = Counter(count_blacks)
  prob_dict = {k: v/sum(count_dict.values()) for k, v in count_dict.items()}
  return prob_dict


def eval_on_data(data,
                 preprocess_fn,
                 params,
                 dist,
                 sess,
                 return_per_pixel=False,
                 dist_family='logistic',
                 wasserstein=False,
                 condition_count=False):
  """predict for data and save log_prob to npy."""

  data_ds = data.map(preprocess_fn).batch(
      params['batch_size']).make_one_shot_iterator()
  data_im = data_ds.get_next()
  
  # NOTE: return_per_pixel collapses channels, e.g. for cifar
  # log_prob is [b, h, w]
  # emd is [b, h, w, c]
  num_zeros = tf.reduce_sum(tf.cast(tf.math.equal(data_im['image'], tf.zeros_like(data_im['image'])), tf.int32), axis=[1, 2, 3])
  # num_zeros = tf.Print(num_zeros, [tf.shape(num_zeros), tf.shape(data_im['image'])], summarize=10, message="num zeros")
  if condition_count:
    log_prob = dist.log_prob(data_im['image'], return_per_pixel=return_per_pixel, dist_family=dist_family, conditional_input=num_zeros)
  else:
    log_prob = dist.log_prob(data_im['image'], return_per_pixel=return_per_pixel, dist_family=dist_family)
  # not accurate for logistic_transform but we don't care
  if dist_family == 'logistic':
    if wasserstein:
      emd = emd_logistic(dist.locs, dist.scales, data_im['image'], agg='conditional')
  elif dist_family in ['categorical', 'logistic_transform', 'normal_transform']:
    if wasserstein:
      import warnings
      warnings.warn("emd for categorical not implemented correctly")
      emd = log_prob
      # raise NotImplemented("Wasserstein not implemented for categorical eval")
  elif dist_family == 'uniform':
    if wasserstein:
      emd = emd(dist.locs, dist.scales, data_im['image'], agg='conditional')
  # log_prob = dist.log_prob(data_im['image'], return_per_pixel=return_per_pixel)
  # image = tf.placeholder(tf.float32, shape=dist.image_shape)
  gradients = tf.gradients(log_prob, data_im['image'])[0]
  grad_norm = tf.norm(tf.reshape(gradients, [tf.shape(gradients)[0], -1]), axis=1)
  # grad_norm = tf.Print(grad_norm, [tf.shape(log_prob), tf.shape(data_im['image']), tf.shape(gradients), tf.shape(grad_norm)], summarize=10)
  log_prob_i_list = []
  label_i_list = []
  image_i_list = []
  grad_i_list = []
  locs_list = []
  scales_list = []
  emd_i_list = []
  count_i_list = []

  # eval on dataset
  while True:
    try:
      label, img = data_im['label'], data_im['image']
      log_prob_np, label_np, image_np, grad_norm_np, locs_np, scales_np, emd_np, count_np = sess.run(
        [log_prob, label, img, grad_norm, dist.locs, dist.scales, emd, num_zeros])
      grad_i_list.append(grad_norm_np)
      # log_prob_i_list.append(np.expand_dims(log_prob_np, axis=-1))
      log_prob_i_list.append(log_prob_np)
      label_i_list += list(label_np.reshape(-1))
      image_i_list.append(image_np)
      locs_list.append(locs_np)
      scales_list.append(scales_np)
      emd_i_list.append(np.expand_dims(emd_np, axis=-1))
      # print(count_np.shape)
      count_i_list.append(count_np)

    except tf.errors.OutOfRangeError:
      print('break')
      break

  grad_i_np = np.hstack(grad_i_list)
  log_prob_i_t_np = np.vstack(log_prob_i_list)
  log_prob_i_np = np.sum(
      log_prob_i_t_np.reshape(log_prob_i_t_np.shape[0], -1), axis=1)
  if condition_count:
    counts = np.concatenate(count_i_list)
    count_to_prob_dict = get_count_dict()
    log_p_count = np.log(np.array([count_to_prob_dict.get(count, 1e-500) for count in counts]))
    # print(log_p_count.shape)
    log_prob_i_np = log_prob_i_np + log_p_count
  emd_i_t_np = np.vstack(emd_i_list)
  emd_i_np = np.sum(
      emd_i_t_np.reshape(emd_i_t_np.shape[0], -1), axis=1)
  print('emd shapes', emd_i_t_np.shape, emd_i_np.shape)
  label_i_np = np.array(label_i_list)
  image_i_np = np.squeeze(np.vstack(image_i_list)).reshape(
      -1, params['n_dim'], params['n_dim'], params['n_channel'])
  out = {
    'log_probs': log_prob_i_np, 'labels': label_i_np, 'images': image_i_np, 'grads': grad_i_np,
    'locs': locs_list, 'scales': scales_list, 'emds': emd_i_np
  }
  # log_prob is [b, h, w]
  # emd is [b, h, w, c]
  if return_per_pixel:
    out['log_probs_per_pixel'] = np.squeeze(log_prob_i_t_np)
    out['emds_per_pixel'] = np.squeeze(emd_i_t_np)
  if condition_count:
    out['log_p_count'] = log_p_count
  return out

def shape_list(x):
    """
    Deal with dynamic shape in tensorflow cleanly
    """
    ps = x.get_shape().as_list()
    ts = tf.shape(input=x)
    return [ts[i] if ps[i] is None else ps[i] for i in range(len(ps))]

def emd_logistic(locs, scales, labels, agg='batch'):
  # grid is [256, b, h, w, c]
  # first axis goes 0-255
  # locs = tf.Print(locs, [tf.shape(locs), tf.shape(scales), tf.shape(labels)], 'shape before', summarize=10)
  b, h, w, c = shape_list(labels)
  grid = tf.reshape(tf.repeat(tf.cast(tf.range(256), tf.float32), b * h * w * c), (256, b, h, w, c))
  # [b, h, w, m, c] -> [1, b, h, w, c]
  locs = tf.reshape(tf.repeat(tf.squeeze(locs, axis=-2), 256), (256, b, h, w, c))
  scales = tf.reshape(tf.repeat(tf.squeeze(scales, axis=-2), 256), (256, b, h, w, c))
  labels = tf.reshape(tf.repeat(labels, 256), (256, b, h, w, c))
  # locs = tf.Print(locs, [tf.shape(locs), tf.shape(scales), tf.shape(labels)], 'locs after', summarize=10)
  probas = (
    tf.where(
      tf.math.equal(grid, 0),
      tf.math.sigmoid(tf.math.divide(0 + .5 - locs,scales)),
      tf.where(
        tf.math.equal(grid, 255),
        1 - tf.math.sigmoid(tf.math.divide(255 - .5 - locs,scales)),
        tf.math.sigmoid(tf.math.divide(grid + .5 - locs, scales)) - tf.math.sigmoid(tf.math.divide(grid - .5 - locs, scales))
      )
    )
  )
  # tf.debugging.assert_near(tf.reduce_sum(probas, axis=0), 1)
  loss = tf.reduce_sum(probas * tf.math.square(grid - labels), axis=0)
  # loss == loss * 1 elementwise?
  tf.debugging.assert_near(loss, loss * tf.reduce_sum(probas, axis=0)).mark_used()
  # loss = tf.Print(loss, [locs, scales, labels, grid, probas, loss], summarize=10)
  if agg=='image':
    return tf.reduce_sum(loss, axis=(1, 2, 3))
  # per_pixel and don't aggregate by channel
  elif agg=='conditional':
    return loss
  elif agg =='batch':
    return tf.reduce_sum(loss)

def emd(mins, maxes, label, penalty=100, norm=1, agg='batch'):
  if norm == 0:
    loss = tf.math.abs(label - tf.math.divide(mins + maxes, 2))
  if norm == 1:
    loss_outside = tf.math.abs(label - tf.math.divide(mins + maxes, 2))
    loss_inside = [tf.math.square(label) - tf.math.multiply(label, (mins + maxes)) + tf.math.divide(tf.math.square(mins) + tf.math.square(maxes), 2)]
    # loss_inside = tf.Print(loss_inside, [tf.shape(loss_inside), tf.shape(loss_outside)], "before", summarize=10)
    loss_inside = tf.math.divide(loss_inside, tf.math.abs(maxes - mins))
    # loss_inside = tf.Print(loss_inside, [tf.shape(loss_inside), tf.shape(loss_outside)], "after", summarize=10)
    loss = tf.where(
      tf.math.logical_and(
        tf.math.less(label, tf.maximum(mins, maxes)), tf.math.greater(label, tf.minimum(mins, maxes))),
      tf.squeeze(loss_inside, 0),
      loss_outside
    )
  elif norm == 2:
    loss = (tf.math.square(label)
     - label * (mins + maxes)
     + (tf.math.square(mins) + tf.math.square(maxes) + tf.multiply(mins, maxes)) / 3
    )

  if agg=='image':
    return tf.reduce_sum(loss, axis=(1, 2, 3))
  # per_pixel and don't aggregate by channel
  elif agg=='conditional':
    # [b, h, w, c]
    return loss
  elif agg =='batch':
    return tf.reduce_sum(loss)