import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def num_initial_zeros(images):
    """ [n, 28 * 28] """
    indices = np.where(images == 1)
    indices_per_row = []
    for i in range(len(images)):
        row_indices, = np.where(indices[0] == i)
        indices_per_row.append(min(indices[1][row_indices]))
    return indices_per_row

if __name__ == "__main__":
    results_dir = 'genomics_ood/images_ood/fashion/mle_lr0.001_p0/expfashion/rescaleFalse/'
    # results_dir = 'genomics_ood/images_ood/fashion_pixel_bin2/expfashion/rescaleFalse/'
    images_in = np.load(os.path.join(results_dir, 'images_in.npy'))
    images_ood = np.load(os.path.join(results_dir, 'images_ood.npy'))
    images_in = images_in.reshape((images_in.shape[0], -1))
    images_ood = images_ood.reshape((images_ood.shape[0], -1))
    # binarize
    images_in[:] = np.where(images_in > 127.5, 1, 0)
    images_ood[:] = np.where(images_ood > 127.5, 1, 0)

    # images_in = images_in[:20]
    # images_ood = images_ood[:20]
    # for images in [images_in, images_ood]:
    #     count = 0
    #     for img in images:
    #         for i in range(len(img)):
    #             if list(img[i:i+21]) == [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]:
    #                 count += 1
    #     print(count)

    # images_in = images_in.reshape((-1, 28, 28))
    # images_ood = images_ood.reshape((-1, 28, 28))
    # for images in [images_in, images_ood]:
    #     count = 0
    #     for image in images:
    #         for row in image:
    #             if sum(row) == 0:
    #                 count += 1
    #     print(count)

    # sns.distplot(num_initial_zeros(images_in), label='images_in')
    # sns.distplot(num_initial_zeros(images_ood), label='images_ood')
    # plt.savefig('num_initial_zeros.png')

    # for n in range(56, 100, 28):
    #     in_combinations = set([tuple(pixels) for pixels in images_in[:, :n]])
    #     ood_combinations = set([tuple(pixels) for pixels in images_ood[:, :n]])
    #     print(len(in_combinations), len(ood_combinations))
    #     # print([i for i in in_combinations if set(i[:-1]) == set([0])])
    #     # import sys; sys.exit()
    #     print(n, len(ood_combinations - in_combinations))
    #     diff = ood_combinations - in_combinations
    #     diff = list(diff)[0]
    #     print([diff[i*28:(i+1)*28] for i in range(n//28)])
    #     print('**')

    log_probs_in = np.load(os.path.join(results_dir, 'log_probs_per_pixel_in.npy'))
    log_probs_ood = np.load(os.path.join(results_dir, 'log_probs_per_pixel_ood.npy'))
    log_probs_in = log_probs_in.reshape((log_probs_in.shape[0], -1))
    log_probs_ood = log_probs_ood.reshape((log_probs_ood.shape[0], -1))
    bins=np.histogram(log_probs_ood, bins=100)[1]
    for images, log_probs, name in zip([images_in, images_ood], [log_probs_in, log_probs_ood], ['zero_probs_in.png', 'zero_probs_ood.png']):
        fig, axes = plt.subplots(5, 5, sharey=True, sharex=True)
        for i, (img, probs) in enumerate(zip(images[:25], log_probs[:25])):
            ax = axes[i//5, i%5]
            ax.hist(probs[img == 0], bins)
            ax.set_yscale('log')
            # ax.set_ylim([0, 1000])
            ax.set_xlim([-200, 0])
        plt.savefig(os.path.join(results_dir, name))
        plt.clf()

    # fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey=True, sharex=True)
    # print(log_probs_in[images_in == 1].mean(), log_probs_in[images_in == 1].std(), log_probs_in[images_in == 1].min(), log_probs_in[images_in == 1].max())
    # print(log_probs_in[images_in == 0].mean(), log_probs_in[images_in == 0].std(), log_probs_in[images_in == 0].min(), log_probs_in[images_in == 0].max())
    # print(log_probs_ood[images_ood == 1].mean(), log_probs_ood[images_ood == 1].std(), log_probs_ood[images_ood == 1].min(), log_probs_ood[images_ood == 1].max())
    # print(log_probs_ood[images_ood == 0].mean(), log_probs_ood[images_ood == 0].std(), log_probs_ood[images_ood == 0].min(), log_probs_ood[images_ood == 0].max())

    # bins=np.histogram(log_probs_ood, bins=100)[1]
    # ax1.hist(log_probs_in[images_in == 1], bins, label='non-zero')
    # ax1.set_yscale('log')
    # ax1.set_title('in-dist non-zero')
    # ax2.hist(log_probs_in[images_in == 0], bins, label='zero')
    # ax2.set_yscale('log')
    # ax2.set_title('in-dist zero')
    # ax3.hist(log_probs_ood[images_ood == 1], bins, label='non-zero')
    # ax3.set_yscale('log')
    # ax3.set_title('out-dist non-zero')
    # ax4.hist(log_probs_ood[images_ood == 0], bins, label='zero')
    # ax4.set_title('out-dist zero')
    # ax4.set_yscale('log')
    # plt.legend()
    # plt.savefig(os.path.join(results_dir, 'log_prob_dists.png'))

    # bins = np.histogram(log_probs_ood.min(axis=1), bins=100)[1]
    # fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
    # ax1.hist(log_probs_in.min(axis=1), bins, label='in-dist')
    # ax2.hist(log_probs_ood.min(axis=1), bins, label='out-dist')
    # ax1.set_title('in-dist')
    # ax2.set_title('out-dist')
    # plt.legend()
    # plt.yscale('log')
    # plt.suptitle('min log prob per image')
    # plt.savefig(os.path.join(results_dir, 'min_log_prob.png'))
    # plt.clf()

    # plt.scatter(log_probs_in.min(axis=1), log_probs_in.sum(axis=1), alpha=.05, label='in-dist')
    # plt.scatter(log_probs_ood.min(axis=1), log_probs_ood.sum(axis=1), alpha=.05, label='out-dist')
    # plt.legend()
    # plt.title('min vs. total log prob per image')
    # plt.savefig(os.path.join(results_dir, 'min_vs_total_log_prob.png'))


    # # what does the out-image look like before the really low proba?
    # indices = np.apply_along_axis(np.argmax, 1, log_probs_ood < -200)
    # assert len(indices) == len(images_ood)
    # fig, axes = plt.subplots(5, 5)
    # i = 0
    # for image, idx in zip(images_ood, indices):
    #     if idx == 0:
    #         continue
    #     if i >= 25:
    #         break
    #     print(idx)
    #     print(image[:idx])
    #     image[idx:] = .5
    #     ax = axes[i // 5, i % 5]
    #     ax.imshow(image.reshape((28, 28)))
    #     i += 1
    # plt.savefig(os.path.join(results_dir, 'image_before_low_prob.png'))