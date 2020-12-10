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

r"""Evaluating OOD detection for generative model-based methods.

    Pure likelihood and LLR.

    Ren, Jie, et al. "Likelihood Ratios for Out-of-Distribution Detection."
    arXiv preprint arXiv:1906.02845 (2019).


"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import flags
import numpy as np
from sklearn.metrics import roc_auc_score
import tensorflow as tf

from genomics_ood import generative
from genomics_ood import utils

TPR_THRES = 0.8

flags.DEFINE_string(
    'model_dir_frgd',
    '/tmp/out_generative/foreground',
    'Directory to ckpts of the foreground generative models.')
flags.DEFINE_string(
    'model_dir_bkgd',
    '/tmp/out_generative/background',
    'Directory to ckpts of the background generative models.')
flags.DEFINE_integer('n_samples', '10000', 'Number of examples for evaluation')
flags.DEFINE_integer('ckpt_step', '900000', 'The step of the selected ckpt.')
flags.DEFINE_string(
    'in_test_data_dir',
    '/tmp/data/after_2016_in_test',
    'Directory to in-distribution test data')
flags.DEFINE_string(
    'ood_test_data_dir',
    '/tmp/data/after_2016_ood_test',
    'Directory to OOD test data')

FLAGS = flags.FLAGS


def list_to_np(list_batch):
  # list_batch is a list of np arrays, each np array is of length batch_size
  return np.stack(list_batch).reshape(-1)


def compute_auc(neg, pos, pos_label=1):
  ys = np.concatenate((np.zeros(len(neg)), np.ones(len(pos))), axis=0)
  neg = np.array(neg)[np.logical_not(np.isnan(neg))]
  pos = np.array(pos)[np.logical_not(np.isnan(pos))]
  scores = np.concatenate((neg, pos), axis=0)
  print(neg.shape, pos.shape, ys.shape, scores.shape)
  auc = roc_auc_score(ys, scores)
  if pos_label == 1:  # label for pos=1 and label for neg=0
    return auc
  else:  # label for pos=0 and label for neg=1
    return 1 - auc


def restore_model_from_ckpt(ckpt_dir, ckpt_file):
  """restore model from ckpt file."""
  # load params
  params_json_file = os.path.join(ckpt_dir, 'params.json')
  params = utils.generate_hparams(params_json_file)
  params.in_val_data_dir = FLAGS.in_test_data_dir
  params.ood_val_data_dir = FLAGS.ood_test_data_dir

  # create model
  tf.reset_default_graph()
  model = generative.SeqModel(params)
  model.reset()
  model.restore_from_ckpt(ckpt_file)
  return params, model

def get_gc_content(x):
  x = np.array(x)
  x = x.reshape((-1, x.shape[-1]))
  mask = (x==1)|(x==2)
  return mask.sum(axis=1)/mask.shape[1]

def main(_):

  model_dir = {'frgd': FLAGS.model_dir_frgd, 'bkgd': FLAGS.model_dir_bkgd}
  # placeholders
  ll_test_in, ll_test_ood = {}, {}
  y_test_in, y_test_ood = {}, {}

  # evaluation on test
  for key in ['frgd', 'bkgd']:

    _, ckpt_file = utils.get_ckpt_at_step(model_dir[key], FLAGS.ckpt_step)
    if not ckpt_file:
      tf.logging.fatal('%s model ckpt not exist', ckpt_file)
    params, model = restore_model_from_ckpt(model_dir[key], ckpt_file)

    # specify test datasets for eval
    params.in_val_file_pattern = 'in_val' # 'in_test'
    params.ood_val_file_pattern = 'ood_val' # 'ood_test'

    (_, in_test_dataset, ood_test_dataset) = generative.load_datasets(
        params, mode_eval=True)

    loss_in_test, _, _, y_in_test, x_in_test = model.pred_from_ckpt(
        in_test_dataset, FLAGS.n_samples)
    loss_ood_test, _, _, y_ood_test, x_ood_test = model.pred_from_ckpt(
        ood_test_dataset, FLAGS.n_samples)

    ll_test_in[key] = -list_to_np(loss_in_test)  # full model likelihood ratio
    ll_test_ood[key] = -list_to_np(loss_ood_test)

    y_test_in[key] = list_to_np(y_in_test)
    y_test_ood[key] = list_to_np(y_ood_test)

    if key == 'frgd':
      gc_content_in = get_gc_content(x_in_test)
      gc_content_ood = get_gc_content(x_ood_test)
    
  # double check if the examples predicted from the foreground model and
  # the background model are in the same order
  assert np.array_equal(y_test_in['frgd'], y_test_in['bkgd'])
  assert np.array_equal(y_test_ood['frgd'], y_test_ood['bkgd'])

  # compute LLR
  llr_test_in = ll_test_in['frgd'] - ll_test_in['bkgd']
  llr_test_ood = ll_test_ood['frgd'] - ll_test_ood['bkgd']

  # compute conditional likelihood
  import joblib as jl
  for bandwidth in [.002]: # [.002, .004, .008, .016, .032]:
    kde = jl.load(f'gc_kde_{bandwidth}.jl')
    ll_gc_in = kde.score_samples(gc_content_in.reshape((-1, 1)))
    ll_gc_ood = kde.score_samples(gc_content_ood.reshape((-1, 1)))
    cl_test_in = ll_test_in['frgd'] - ll_gc_in
    cl_test_ood = ll_test_ood['frgd'] - ll_gc_ood
    cl_test_in = np.nan_to_num(cl_test_in, nan=-9999, posinf=-9999)
    cl_test_ood = np.nan_to_num(cl_test_ood, nan=-9999, posinf=-9999)
    auc_cl = compute_auc(cl_test_in, cl_test_ood, pos_label=0)
    print(bandwidth, auc_cl)
    np.save('ll_gc_in', ll_gc_in)
    np.save('ll_gc_ood', ll_gc_ood)
    np.save('ll_test_in', ll_test_in['frgd'])
    np.save('ll_test_ood', ll_test_ood['frgd'])
    np.save('gc_content_in', gc_content_in)
    np.save('gc_content_ood', gc_content_ood)

  # eval for AUC
  auc_ll = compute_auc(ll_test_in['frgd'], ll_test_ood['frgd'], pos_label=0)
  auc_llr = compute_auc(llr_test_in, llr_test_ood, pos_label=0)
  

  print('AUCs for raw likelihood and likelihood ratio: %s, %s, %s' %
        (auc_ll, auc_llr, auc_cl))
  with open('results.txt', 'a') as f:
    f.write('auc: ' + str(auc_cl) + '\n')


if __name__ == '__main__':
  tf.app.run()
