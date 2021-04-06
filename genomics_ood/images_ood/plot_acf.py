import matplotlib.pyplot as plt
import numpy as np

def autocorr1(x,lags):
    '''numpy.corrcoef, partial'''
    corr=[1. if l==0 else np.corrcoef(x[l:],x[:-l])[0][1] for l in lags]
    return np.array(corr)

if __name__ == "__main__":
    model_path = 'genomics_ood/images_ood/cifar/mle_lr0.0001_p0/expcifar/rescaleTrue/samples.npy'
    # model_path = '../pixel-cnn/cifar_save_dir/cifar_sample0.npz'
    # model_path = 'genomics_ood/images_ood/fashion_pixel/expfashion/rescaleFalse/samples.npy'
    # model_path = 'genomics_ood/images_ood/fashion/mle_lr0.001_p0/expfashion/rescaleFalse/samples.npy'
    in_path = model_path.replace('samples.npy', 'images_in.npy')
    ood_path = model_path.replace('samples.npy', 'images_ood.npy')
    paths = [model_path, in_path, ood_path]
    # names = ['samples_acf.png', 'in_acf.png', 'ood_acf.png']
    names = ['samples', 'in', 'ood']
    means = []
    stds = []
    for path, name in zip(paths, names):
        model_samples = np.load(path)
        if '.npz' in path:
            model_samples = np.concatenate([model_samples[key] for key in sorted(model_samples.files)], axis=0)
            print(model_samples.shape)
        # model_samples = model_samples[:500]
        model_samples = model_samples.reshape((model_samples.shape[0], -1))
        acfs = np.apply_along_axis(lambda x: autocorr1(x, range(2)), 1, model_samples)
        # plt.bar(x=range(acfs.shape[1]), height=np.mean(acfs, axis=0), yerr=np.std(acfs, axis=0))
        # if '.npz' in model_path:
        #     plt.savefig(model_path.replace('cifar_sample0.npz', name))
        # else:
        #     plt.savefig(model_path.replace('samples.npy', name))
        # plt.clf()

        if 'samples' in path:
            continue
        # acfs = acfs[:, 1]
        # log_probs = np.load(model_path.replace('samples.npy', f'log_probs_{name}.npy'))
        # print(acfs.shape, log_probs.shape)
        # plt.scatter(acfs, log_probs, alpha=.2)
        # plt.savefig(model_path.replace('samples.npy', f'{name}_p_vs_corr.png'))


        pixel_deltas = model_samples[:, 1:] - model_samples[:, :-1]
        log_probs = np.load(model_path.replace('samples.npy', f'log_probs_per_pixel_{name}.npy'))
        # don't need first number
        log_probs = log_probs.reshape((log_probs.shape[0], -1))[:, 1:]
        print(pixel_deltas.shape, log_probs.shape)
        plt.scatter(pixel_deltas[1:5].flatten(), log_probs[1:5].flatten(), alpha=.05)
        plt.savefig(model_path.replace('samples.npy', f'{name}_p_vs_corr.png'))


        # log_probs = np.load(model_path.replace('samples.npy', f'log_probs_per_pixel_{name}.npy'))
        # # don't need first number
        # log_probs = log_probs.reshape((log_probs.shape[0], -1))
        # # plt.scatter(model_samples[1:5].flatten(), log_probs[1:5].flatten(), alpha=.05)
        # # plt.savefig(model_path.replace('samples.npy', f'{name}_p_vs_corr.png'))
        # model_samples = model_samples.reshape((model_samples.shape[0], -1))
        # boundaries = log_probs[(model_samples==0)|(model_samples==255)]
        # nonboundaries = log_probs[(model_samples!=0)|(model_samples!=255)]
        # plt.bar(x=range(2), height=[boundaries.mean(), nonboundaries.mean()], yerr=[boundaries.std(), nonboundaries.std()])
        # plt.savefig(model_path.replace('samples.npy', f'{name}_boundaries_vs_middle.png'))


    #     # lag-1
    #     acfs = acfs[:, 1]
    #     mean_acf = np.mean(acfs)
    #     std_acf = np.std(acfs)
    #     means.append(mean_acf)
    #     stds.append(std_acf)
    # plt.bar(x=range(3), height=means, yerr=stds)
    # plt.xticks(range(3), ['samples', 'in', 'ood'])
    # if '.npz' in model_path:
    #     plt.savefig(model_path.replace('cifar_sample0.npz', 'lag1_acf.png'))
    # else:
    #     plt.savefig(model_path.replace('samples.npy', 'lag1_acf.png'))
    