import os
import joblib as jl
import numpy as np
from sklearn.neighbors import KernelDensity
from genomics_ood import generative
from genomics_ood import utils

if __name__ == "__main__":
    ckpt_dir = '../outputs/generative_l250_bs100_lr0.0005_hr30_nrFalse_regl2_regw0.000000_fi-1_mt0.00/model/'
    params_json_file = os.path.join(ckpt_dir, 'params.json')
    params = utils.generate_hparams(params_json_file)
    params.in_val_data_dir = '../data/before_2011_in_tr/'
    params.ood_val_data_dir = '../data/between_2011-2016_ood_val/'

    # specify test datasets for eval
    params.in_val_file_pattern = 'in_tr'
    params.ood_val_file_pattern = 'ood_val'

    (_, in_dataset, _) = generative.load_datasets(
        params, mode_eval=True)
    model = generative.SeqModel(params)
    model.reset()
    test_dataset = in_dataset.batch(model._params.batch_size)
    test_iterator = test_dataset.make_one_shot_iterator()
    model.test_handle = model.sess.run(test_iterator.string_handle())
    x_test = []
    num_samples = 100000
    for _ in range(num_samples // model._params.batch_size):
        out = model.sess.run(
          [model.x],
          feed_dict={
              model.handle: model.test_handle,
              model.dropout_rate: 0
          })
        x_test.append(out[0])
    x = np.array(x_test)
    x = x.reshape((-1, x.shape[-1]))
    KernelDensity(kernel='epanechnikov', bandwidth=0.008)
    kde.fit(x)
    jl.dump('gc_kde.jl', kde)