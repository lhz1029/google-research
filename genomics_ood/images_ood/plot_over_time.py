import os
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    results_dir = 'genomics_ood/images_ood/cifar/mle_lr0.0001_p0/expcifar/rescaleTrue/'
    in_path = '../data/cifar10_test.npy'
    ood_path = '../data/svhn_cropped_test.npy'
    # results_dir = 'genomics_ood/images_ood/fashion/mle_lr0.001_p0/expfashion/rescaleFalse/'
    # in_path = '../data/fashion_mnist_test.npy'
    # ood_path = '../data/mnist_test.npy'

    lp_in = np.load(os.path.join(results_dir, 'log_probs_per_pixel_in.npy'))
    lp_ood = np.load(os.path.join(results_dir, 'log_probs_per_pixel_ood.npy'))
    lp_in = lp_in.reshape((lp_in.shape[0], -1))
    lp_ood = lp_ood.reshape((lp_ood.shape[0], -1))

    plt.bar(range(lp_in.shape[1]), lp_in.mean(axis=0))
    plt.savefig(os.path.join(results_dir, 'lp_over_time_in.png'))
    plt.clf()

    plt.bar(range(lp_ood.shape[1]), lp_ood.mean(axis=0))
    plt.savefig(os.path.join(results_dir, 'lp_over_time_ood.png'))