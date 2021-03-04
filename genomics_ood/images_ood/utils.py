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


def load_tfdata_from_np(np_file, flip=None):
  try:
    with tf.compat.v1.gfile.Open(np_file, mode='rb') as f:
      images = np.load(f)
      labels = np.load(f)
  except ValueError:
    print('tf open failed')
    images = np.load(np_file)
    if flip == 'v':
      images = np.array([np.flipud(img) for img in images])
    elif flip == 'h':
      images = np.array([np.fliplr(img) for img in images])
    labels = images
  dataset = tf.compat.v1.data.Dataset.from_tensor_slices(
      (images, labels)).map(tensor_slices_preprocess)
  return dataset

def load_fmnist_datasets(data_dir, out_data='mnist'):
  """Load fashionMNIST and MNIST dataset from np array."""
  tr_in = load_tfdata_from_np(os.path.join(data_dir, 'fashion_mnist_train.npy'))
  val_in = load_tfdata_from_np(os.path.join(data_dir, 'fashion_mnist_val.npy'))
  test_in = load_tfdata_from_np(
      os.path.join(data_dir, 'fashion_mnist_test.npy'))
  test1_in = load_tfdata_from_np(os.path.join(data_dir, 'fashion_mnist_test1.npy'))

  val_ood = load_tfdata_from_np(os.path.join(data_dir, 'notmnist.npy'))
  test1_ood = load_tfdata_from_np(os.path.join(data_dir, 'mnist_test1.npy'))
  if out_data == 'mnist':
    test_ood = load_tfdata_from_np(os.path.join(data_dir, 'mnist_test.npy'))
  elif out_data == 'hflip':
    test_ood = load_tfdata_from_np(
      os.path.join(data_dir, 'fashion_mnist_test.npy'), flip='h')
  elif out_data == 'vflip':
    test_ood = load_tfdata_from_np(
      os.path.join(data_dir, 'fashion_mnist_test.npy'), flip='v')
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
  return {
      'tr_in': tr_in,
      'val_in': val_in,
      'test_in': test_in,
      'test1_in': test1_in,
      'val_ood': val_ood,
      'test_ood': test_ood,
      'test1_ood': test1_ood
  }

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
    test_ood = load_tfdata_from_np(
        os.path.join(data_dir, 'svhn_cropped_test.npy'))
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
      images = data.reshape((10000, 32, 32, 3))
      labels = images
    test_ood = tf.compat.v1.data.Dataset.from_tensor_slices(
    (images, labels)).map(tensor_slices_preprocess)
  elif out_data == 'imagenet32':
    raise NotImplemented("imagenet32 not yet set up")
  return {
      'tr_in': tr_in,
      'val_in': val_in,
      'test_in': test_in,
      'test_ood': test_ood  # val_ood is val_in_grey
  }

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
  np.save('neg.npy', neg)
  np.save('pos.npy', pos)
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


def eval_on_data(data,
                 preprocess_fn,
                 params,
                 dist,
                 sess,
                 return_per_pixel=False,
                 wasserstein=False):
  """predict for data and save log_prob to npy."""

  data_ds = data.map(preprocess_fn).batch(
      params['batch_size']).make_one_shot_iterator()
  data_im = data_ds.get_next()

  if wasserstein:
    log_prob = dist.log_prob(data_im['image'], return_per_pixel=return_per_pixel, dist_family='uniform')
  else:  
    log_prob = dist.log_prob(data_im['image'], return_per_pixel=return_per_pixel)
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

  # eval on dataset
  while True:
    try:
      label, img = data_im['label'], data_im['image']
      log_prob_np, label_np, image_np, grad_norm_np, locs_np, scales_np = sess.run(
        [log_prob, label, img, grad_norm, dist.locs, dist.scales])
      grad_i_list.append(grad_norm_np)
      log_prob_i_list.append(np.expand_dims(log_prob_np, axis=-1))
      label_i_list += list(label_np.reshape(-1))
      image_i_list.append(image_np)
      locs_list.append(locs_np)
      scales_list.append(scales_np)

    except tf.errors.OutOfRangeError:
      print('break')
      break

  grad_i_np = np.hstack(grad_i_list)
  log_prob_i_t_np = np.vstack(log_prob_i_list)
  log_prob_i_np = np.sum(
      log_prob_i_t_np.reshape(log_prob_i_t_np.shape[0], -1), axis=1)
  label_i_np = np.array(label_i_list)
  image_i_np = np.squeeze(np.vstack(image_i_list)).reshape(
      -1, params['n_dim'], params['n_dim'], params['n_channel'])
  out = {
    'log_probs': log_prob_i_np, 'labels': label_i_np, 'images': image_i_np, 'grads': grad_i_np,
    'locs': locs_list, 'scales': scales_list
  }
  if return_per_pixel:
    out['log_probs_per_pixel'] = np.squeeze(log_prob_i_t_np)
  return out

def emd(mins, maxes, label, penalty=100, norm=1, per_image=False):
  # if norm == 1:
  #   loss = tf.math.abs(mins - label) + tf.math.abs(maxes - label)
  # elif norm == 2:
  #   loss = tf.math.square(mins - label) + tf.math.square(maxes - label)
  # # return tf.reduce_sum(loss)
  # penalty_mask = tf.where(tf.math.logical_and(
  #   tf.math.less(label, tf.maximum(mins, maxes)), tf.math.greater(label, tf.minimum(mins, maxes))
  # ), tf.zeros_like(label), tf.ones_like(label))
  # return tf.reduce_sum(loss + penalty * penalty_mask)
  if norm == 0:
    loss = tf.math.abs(label - tf.math.divide(mins + maxes, 2))
  if norm == 1:
    # loss_outside = tf.math.abs(mins - label) + tf.math.abs(maxes - label)
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
    # loss = [tf.math.square(label) * tf.math.abs(maxes - mins)
    #  - label * tf.math.abs(tf.math.square(mins) - tf.math.square(maxes))
    #  + tf.math.abs(tf.pow(mins, tf.constant([3.])) - tf.pow(maxes, tf.constant([3.])))
    # ]
    # loss = tf.math.divide(loss, tf.math.abs(maxes - mins))
    loss = (tf.math.square(label)
     - label * (mins + maxes)
     + (tf.math.square(mins) + tf.math.square(maxes) + tf.multiply(mins, maxes)) / 3
    )
  if per_image:
    return tf.reduce_sum(loss, axis=(1, 2, 3))
  else:
    return tf.reduce_sum(loss)