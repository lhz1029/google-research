import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from collections import defaultdict
from matplotlib.colors import LogNorm
import matplotlib
import os

import torch
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal
from torch.distributions.transforms import SigmoidTransform, AffineTransform
from torch.distributions.transformed_distribution import TransformedDistribution


def sigmoid(x):
    try:
        if x < -700: return 1
        return 1/(1+math.exp(-x))
    except OverflowError:
        print(x)
        import traceback; traceback.print_exc()
        import sys; sys.exit()
        

def discretized_logistic_prob(x, mean, scale):
    return sigmoid((x + .5 - mean)/scale) - sigmoid((x - .5 - mean)/scale)

def discretized_logistic(mean, scale, high=255):
    probs = []
    probs.append(sigmoid((0 + .5 - mean)/scale))
    for x in range(1, high):
        probs.append(discretized_logistic_prob(x, mean, scale))
    probs.append(1 - sigmoid((high - .5 - mean)/scale))
    return probs

def logistic_transform(mean, scale, high=255):
    base_distribution = Uniform(0, 1)
    transforms = [SigmoidTransform().inv, AffineTransform(loc=mean, scale=scale)]
    logistic = TransformedDistribution(base_distribution, transforms)
    dist = TransformedDistribution(logistic, [SigmoidTransform(), AffineTransform(loc=0, scale=high)])
    probs = [dist.cdf(torch.tensor(x + 1.)) - dist.cdf(torch.tensor(float(x))) for x in range(256)]
    return probs

if __name__ == "__main__":
    # cifar = np.load('../data/fashion_mnist_test.npy')
    # svhn = np.load('../data/mnist_test.npy')
    path = 'genomics_ood/images_ood/fashion/mle_lr0.001_p0/expfashion/rescaleFalse/'
    # path = 'genomics_ood/images_ood/cifar/mle_lr0.0001_p0/expcifar/rescaleTrue/'
    # path = 'genomics_ood/images_ood/fashion2/mle_lr0.001_m1_lt_v6/expfashion/rescaleFalse/'
    # locs = np.load(os.path.join(path, 'locs_in.npy'), allow_pickle=True)
    # scales = np.load(os.path.join(path, 'scales_in.npy'), allow_pickle=True)
    cifar = np.load(os.path.join(path, 'images_in.npy'), allow_pickle=True)
    svhn = np.load(os.path.join(path, 'images_ood.npy'), allow_pickle=True)

    # bad
    # cifar = np.load('../data/cifar10_test.npy')
    # svhn = np.load('../data/svhn_cropped_test.npy')
    # locs = np.load('genomics_ood/images_ood/cifar/mle_lr0.0001_p0/expcifar/rescaleTrue/locs_in.npy', allow_pickle=True)
    # scales = np.load('genomics_ood/images_ood/cifar/mle_lr0.0001_p0/expcifar/rescaleTrue/scales_in.npy', allow_pickle=True)

    # locs = np.concatenate(locs)
    # scales = np.concatenate(scales)
    # # remove mixture number
    # locs = np.squeeze(locs, axis=-2)
    # scales = np.squeeze(scales, axis=-2)
    # assert len(cifar) == len(locs) == len(scales)
    # assert len(locs.shape) == len(scales.shape) == 4, "{}".format(locs.shape)
    # locs = locs.reshape((locs.shape[0], -1))
    # scales = scales.reshape((scales.shape[0], -1))
    # # print(locs.shape, scales.shape)

    cifar_by_first_pixel = defaultdict(list)

    # # learned dist
    # for i, img in enumerate(cifar):
    #     cifar_by_first_pixel[img[0][0][0]].append(i)
    
    # conditional_probs = []
    # for i in reversed(range(256)):
    #     idx = cifar_by_first_pixel[i]
    #     relevant_locs = locs[idx]
    #     relevant_scales = scales[idx]
    #     if relevant_scales.size == 0:
    #         probs = np.zeros(256) + 1e-18
    #     else:
    #         # print(relevant_locs[:, 1])
    #         # print(relevant_scales[:, 1])
    #         probs = discretized_logistic(relevant_locs[0, 1], relevant_scales[0, 1], high=255)
    #     conditional_probs.append(probs)
    # # row is p(x_t | x_{<t}), i.e. the conditional distribution
    # conditional_probs = np.array(conditional_probs)
    # conditional_probs = np.maximum(conditional_probs, np.zeros_like(conditional_probs) + 1e-18)
    # print('conditional', np.array(conditional_probs).shape)
    # print(np.array(conditional_probs).min())
    # print(np.array(conditional_probs[0:-1]).max())
    # im = plt.imshow(conditional_probs, cmap='hot', norm=LogNorm(vmin=conditional_probs.min(), vmax=conditional_probs.max()))
    # current_cmap = matplotlib.cm.get_cmap()
    # current_cmap.set_bad(color='blue')
    # plt.colorbar(im)
    # plt.savefig(os.path.join(path, 'model_second_pixel.png'))

    # # # empirical dist
    # cifar_by_first_pixel = defaultdict(list)
    # for i, img in enumerate(cifar):
    #     img = img.reshape(-1)
    #     cifar_by_first_pixel[img[0]].append(img[1])
    # conditional_probs = []
    # for i in reversed(range(256)):
    #     indices = cifar_by_first_pixel[i]
    #     counts = np.zeros(256)
    #     for idx in indices:
    #         counts[int(idx)] += 1
    #     counts = counts / sum(counts)
    #     # assert sum(counts) == 1, sum(counts)
    #     conditional_probs.append(counts)
    # conditional_probs = np.array(conditional_probs)
    # print(np.array(conditional_probs[0:-1]).min())
    # im = plt.imshow(conditional_probs, cmap='hot') #, norm=LogNorm(vmin=conditional_probs.min(), vmax=conditional_probs.max()))
    # plt.colorbar(im)
    # plt.savefig(os.path.join(path, 'empirical_second_pixel_no_log.png'))
    # plt.clf()
    # conditional_probs = np.maximum(conditional_probs, np.zeros_like(conditional_probs) + 1e-18)
    # im = plt.imshow(conditional_probs, cmap='hot', norm=LogNorm(vmin=conditional_probs.min(), vmax=conditional_probs.max()))
    # plt.colorbar(im)
    # plt.savefig(os.path.join(path, 'empirical_second_pixel.png'))
    
    # avg conditional
    # all_probs = [discretized_logistic(loc, scale) for loc, scale in zip(locs.flatten()[:200000], scales.flatten()[:200000])]
    # all_probs = np.array(all_probs)
    # (print(all_probs.shape))
    # plt.bar(range(256), all_probs.mean(axis=0))  #, yerr=all_probs.std(axis=0))
    # # plt.savefig('genomics_ood/images_ood/cifar/mle_lr0.0001_p0/expcifar/rescaleTrue/avg_conditional_in.png')
    # plt.yscale('log')
    # plt.savefig('genomics_ood/images_ood/fashion/mle_lr0.001_p0/expfashion/rescaleFalse/avg_conditional_in.png')

    # plt.clf()
    # plt.hist(cifar.flatten(), bins=range(256))
    # plt.savefig('genomics_ood/images_ood/cifar/mle_lr0.0001_p0/expcifar/rescaleTrue/avg_probs_in.png')
    # # plt.yscale('log')
    # # plt.savefig('genomics_ood/images_ood/fashion/mle_lr0.001_p0/expfashion/rescaleFalse/avg_probs_in.png')

    # plt.clf()
    # plt.hist(svhn.flatten(), bins=range(256))
    # plt.savefig('genomics_ood/images_ood/cifar/mle_lr0.0001_p0/expcifar/rescaleTrue/avg_probs_ood.png')
    # # plt.yscale('log')
    # # plt.savefig('genomics_ood/images_ood/fashion/mle_lr0.001_p0/expfashion/rescaleFalse/avg_probs_ood.png')


    # # conditional when value is boundary vs not
    locs_ood = np.load(os.path.join(path, 'locs_ood.npy'), allow_pickle=True)
    scales_ood = np.load(os.path.join(path, 'scales_ood.npy'), allow_pickle=True)
    locs_ood = np.concatenate(locs_ood, axis=0)
    scales_ood = np.concatenate(scales_ood, axis=0)
    locs_ood = locs_ood.reshape((locs_ood.shape[0], -1))
    scales_ood = scales_ood.reshape((scales_ood.shape[0], -1))

    locs_in = np.load(os.path.join(path, 'locs_in.npy'), allow_pickle=True)
    scales_in = np.load(os.path.join(path, 'scales_in.npy'), allow_pickle=True)
    locs_in = np.concatenate(locs_in, axis=0)
    scales_in = np.concatenate(scales_in, axis=0)
    locs_in = locs_in.reshape((locs_in.shape[0], -1))
    scales_in = scales_in.reshape((scales_in.shape[0], -1))

    # images_in = cifar.reshape((cifar.shape[0], -1))
    # images_ood = svhn.reshape((svhn.shape[0], -1))
    images_in = cifar.flatten()
    images_ood = svhn.flatten()
    # num_pixels = 3

    locs = locs_ood
    scales = scales_ood
    images = images_ood
    # # all_probs = [discretized_logistic(loc, scale) for loc, scale in zip(locs.flatten()[:num_pixels], scales.flatten()[:num_pixels])]
    # all_probs = [logistic_transform(loc, scale) for loc, scale in zip(locs.flatten()[:num_pixels], scales.flatten()[:num_pixels])]
    # all_probs = np.array(all_probs)
    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    # probs1 = all_probs[images[:num_pixels] == 0]
    # ax1.bar(range(probs1.shape[1]), probs1.mean(axis=0))
    # print(probs1.mean(axis=0).max(), probs1, probs1.shape)
    # ax1.set_title('zero boundary')
    # probs2 = all_probs[images[:num_pixels] == 255]
    # ax2.bar(range(probs2.shape[1]), probs2.mean(axis=0))
    # print(probs2.mean(axis=0).max(), probs2, probs2.shape)
    # ax2.set_title('255 boundary')
    # probs3 = all_probs[(images[:num_pixels] > 0) & (images[:num_pixels] < 255)]
    # ax3.bar(range(probs3.shape[1]), probs3.mean(axis=0))
    # print(probs3.mean(axis=0).max(), probs3, probs3.shape)
    # ax3.set_title('middle')
    # plt.savefig(os.path.join(path, 'probs_boundary_vs_middle_ood.png'))


    # locs = locs_in
    # scales = scales_in
    # images = images_in
    num_pixels = 1000
    mask = (images > 0) & (images < 255)
    # mask = images == 255
    # middle pixel dist
    all_probs = [discretized_logistic(loc, scale) for loc, scale in zip(locs.flatten()[mask][:num_pixels], scales.flatten()[mask][:num_pixels])]
    # all_probs = [logistic_transform(loc, scale) for loc, scale in zip(locs.flatten()[mask][:num_pixels], scales.flatten()[mask][:num_pixels])]
    all_probs = np.array(all_probs)
    plt.bar(range(all_probs.shape[1]), all_probs.mean(axis=0))
    plt.savefig(os.path.join(path, 'middle_ood_lp.png'))

    # log_probs_per_pixel_in = np.load(os.path.join(path, 'log_probs_per_pixel_in.npy'), allow_pickle=True)
    # log_probs_per_pixel_ood = np.load(os.path.join(path, 'log_probs_per_pixel_ood.npy'), allow_pickle=True)
    # log_probs_per_pixel_in = np.concatenate(log_probs_per_pixel_in, axis=0).flatten()
    # log_probs_per_pixel_ood = np.concatenate(log_probs_per_pixel_ood, axis=0).flatten()
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # ax1.scatter(images_in[:num_pixels], log_probs_per_pixel_in[:num_pixels], alpha=.001)
    # ax1.set_title('in-dist')
    # ax2.scatter(images_ood[:num_pixels], log_probs_per_pixel_ood[:num_pixels], alpha=.001)
    # ax2.set_title('out-dist')
    # plt.savefig(os.path.join(path, 'log_prob_vs_pixel.png'))

    # print(np.quantile(log_probs_per_pixel_in[images_in == 0], [0, .1, .25, .5, .75, .9, 1]))
    # print(np.quantile(log_probs_per_pixel_in[images_in == 255], [0, .1, .25, .5, .75, .9, 1]))
    # print(np.quantile(log_probs_per_pixel_in[(images_in > 0)&(images_in < 255)], [0, .1, .25, .5, .75, .9, 1]))

    # print(np.quantile(log_probs_per_pixel_ood[images_ood == 0], [0, .1, .25, .5, .75, .9, 1]))
    # print(np.quantile(log_probs_per_pixel_ood[images_ood == 255], [0, .1, .25, .5, .75, .9, 1]))
    # print(np.quantile(log_probs_per_pixel_ood[(images_ood > 0)&(images_ood < 255)], [0, .1, .25, .5, .75, .9, 1]))