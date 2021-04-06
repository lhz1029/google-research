import glob
import itertools

fashion_exps = ['fashion-mnist',
    # 'fashion-vflip',
    # 'fashion-hflip',
    'fashion-omniglot',
    # 'fashion-gaussian',
    # 'fashion-uniform',
]
cifar_exps = ['cifar-svhn',
    'cifar-celeba',
    # 'cifar-vflip',
    # 'cifar-hflip',
    # 'cifar-gaussian',
    # 'cifar-uniform',
    'cifar-cifar100',
    # 'cifar-imagenet'
]

fashion_dirs = glob.glob('genomics_ood/images_ood/fashion/*.001*/expfashion/rescaleFalse')
cifar_dirs = glob.glob('genomics_ood/images_ood/cifar/*.0001*/expcifar/rescaleTrue')

commands = []
fashion_dirs = [d for d in fashion_dirs if 'mle' in d]
for dir, exp in itertools.product(fashion_dirs, fashion_exps):
    command = f"python -m genomics_ood.images_ood.eval_wasserstein --exp={exp} --model_dir={dir} --ckpt_step=50000 --repeat_id=0"
    # bad mle
    # command = f"python -m genomics_ood.images_ood.eval_wasserstein --exp={exp} --model_dir={dir} --ckpt_step=1000 --repeat_id=0"
    if 'mle' in dir:
        command += ' --logistic'
    commands.append(command)

for dir, exp in itertools.product(cifar_dirs, cifar_exps):
    command = f"python -m genomics_ood.images_ood.eval_wasserstein --exp={exp} --model_dir={dir} --ckpt_step=600000 --repeat_id=0"
    if 'mle' in dir:
        command += ' --logistic'
    commands.append(command)

with open("jobs_ood_eval.txt", "w") as f:
    f.write('\n'.join(commands) + '\n')