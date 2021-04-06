import numpy as np
import matplotlib.pyplot as plt
import math
import argparse
import math


def sigmoid(x):
    return 1/(1+math.exp(-x))

def discretized_logistic_prob(x, mean, scale):
    if x == 0:
        return sigmoid((0 + .5 - mean)/scale)
    elif x == 255:
        return 1 - sigmoid((255 - .5 - mean)/scale)
    else:
        return sigmoid((x + .5 - mean)/scale) - sigmoid((x - .5 - mean)/scale)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Results summary')
    parser.add_argument('-l', '--locs_path', default='', type=str)
    args = parser.parse_args()

    locs = np.load(args.locs_path, allow_pickle=True)
    scales = np.load(args.locs_path.replace('locs', 'scales'), allow_pickle=True)
    log_probs = np.load(args.locs_path.replace('locs', 'log_probs_per_pixel'), allow_pickle=True)
    imgs = np.load(args.locs_path.replace('locs', 'images'), allow_pickle=True)
    if 'bin2' in args.locs_path:
        imgs[:] = np.where(imgs > 127.5, 255, 0)
    elif 'bin' in args.locs_path:
        imgs[:] = np.where(imgs > 127.5, 1, 0)
    # if 'fashion' in args.locs_path and '_in' in args.locs_path:
    #     exp = 'fashion'
    #     imgs = np.load('../data/fashion_mnist_test.npy')
    # elif 'fashion' in args.locs_path and '_ood' in args.locs_path:
    #     exp = 'mnist'
    #     imgs = np.load('../data/mnist_test.npy')
    # elif 'cifar' in args.locs_path and '_in' in args.locs_path:
    #     exp = 'cifar'
    #     imgs = np.load('../data/cifar10_test.npy')
    # elif 'cifar' in args.locs_path and '_train' in args.locs_path:
    #     exp = 'cifar'
    #     imgs = np.load('../data/cifar10_train.npy')
    # elif 'cifar' in args.locs_path and '_ood' in args.locs_path:
    #     exp = 'svhn'
    #     imgs = np.load('../data/svhn_cropped_test.npy')
    num_cols = 3
    locs = np.concatenate(locs, axis=0)
    scales = np.concatenate(scales, axis=0)
    fig, axes = plt.subplots(num_cols, num_cols, sharey=True, sharex=True, figsize=(16, 10))
    for i in range(num_cols ** 2):
        # j = i
        j = i * 28
        # ax2 = axes[i//num_cols, i%num_cols].twinx()
        # ax2.plot(log_probs[j].reshape(-1), label='log_probs', color='purple')
        if 'w_' in args.locs_path:
            mins = np.minimum(locs[j].reshape(-1), scales[j].reshape(-1))
            maxes = np.maximum(locs[j].reshape(-1), scales[j].reshape(-1))
            axes[i//num_cols, i%num_cols].plot(mins, label='mins')
            axes[i//num_cols, i%num_cols].plot(maxes, label='maxes')
            axes[i//num_cols, i%num_cols].plot(mins - imgs[j].flatten(), label='distance_from_pixel (min boundary)', color='lime')
            axes[i//num_cols, i%num_cols].plot(maxes - imgs[j].flatten(), label='distance_from_pixel (max boundary)', color='skyblue')
            axes[i//num_cols, i%num_cols].plot(log_probs[j].reshape(-1) * 20, label='log_probs * 20', color='purple')
        else:
            if 'cifar' in args.locs_path:
                # axes[i//num_cols, i%num_cols].plot(locs[j].reshape(-1) - imgs[j].flatten(), label='distance_from_pixel', color='lime')
                log_probs = [discretized_logistic_prob(pixel, loc, scale) for pixel, loc, scale in zip(imgs.flatten(), locs[j].flatten(), scales[j].flatten())]
                log_probs = np.array(log_probs)
                # axes[i//num_cols, i%num_cols].plot(log_probs * 100, label='log_probs * 100', color='purple')
                axes[i//num_cols, i%num_cols].plot(log_probs, label='log_probs', color='purple')
            if 'fashion' in args.locs_path:
                axes[i//num_cols, i%num_cols].plot(log_probs[j].reshape(-1) * 1000, label='log_probs * 1000', color='purple')
                # axes[i//num_cols, i%num_cols].plot(locs[j].reshape(-1) - imgs[j].flatten(), label='distance_from_pixel', color='lime')
                axes[i//num_cols, i%num_cols].plot(imgs[j].flatten(), label='pixel', color='lime')
                # emds = np.load(args.locs_path.replace('locs', 'emds_per_pixel'), allow_pickle=True)
                # axes[i//num_cols, i%num_cols].plot(emds[j].reshape(-1), label='emds', color='skyblue')
                # emds = np.load(args.locs_path.replace('locs_in', 'emd_cont_in_2').replace('locs_ood', 'emd_cont_ood_2'), allow_pickle=True)
                # axes[i//num_cols, i%num_cols].plot(emds[j].reshape(-1), label='emds_cont', color='pink')
                # emds = np.load(args.locs_path.replace('locs_in', 'emd_cont_in_bounded_2').replace('locs_ood', 'emd_cont_ood_bounded_2'), allow_pickle=True)
                # axes[i//num_cols, i%num_cols].plot(emds[j].reshape(-1), label='emds_cont_bounded', color='maroon')
                # emds = np.load(args.locs_path.replace('locs', 'emds_d1kb_per_pixel'), allow_pickle=True)
                # axes[i//num_cols, i%num_cols].plot(emds[j].reshape(-1), label='emds_d1kb', color='gray')
                plt.ylim([-500, 250])
            axes[i//num_cols, i%num_cols].plot(locs[j].reshape(-1), label='locs')
            axes[i//num_cols, i%num_cols].plot(scales[j].reshape(-1), label='scales')
        
    fig.tight_layout()
    plt.legend()
    plt.savefig(args.locs_path.replace('locs', 'pixel_dists').replace('.npy', '.png'))

    plt.clf()
    locs = np.concatenate(locs, axis=0)
    scales = np.concatenate(scales, axis=0)
    imgs = np.concatenate(imgs, axis=0)
    plt.scatter(scales.flatten()[:10000], locs.flatten()[:10000], label='locs vs scales', alpha=.2)
    plt.scatter(scales.flatten()[:10000], imgs.flatten()[:10000], label='pixel vs scales', alpha=.2)
    plt.title('locs vs. scales')
    plt.savefig(args.locs_path.replace('locs', 'locs_vs_scales').replace('.npy', '.png'))

    plt.clf()
    plt.scatter(scales.flatten()[:10000], log_probs.flatten()[:10000], alpha=.2)
    plt.savefig(args.locs_path.replace('locs', 'lp_vs_scales').replace('.npy', '.png'))