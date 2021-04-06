commands = []

# # Wasserstein
# for p in [0]: #, 100, 1000]: #, 10000]:
#     for lr in [.001, .0001]:  #[.1, .01, .001, .0001, .00001]:
#         for n in [0, 1, 2]:  #[1, 2]:
#             commands.append(f"""python -m genomics_ood.images_ood.train --exp=cifar --data_dir=../data --out_dir=genomics_ood/images_ood/cifar/w_lr{lr}_p{p}_n{n} --total_steps=600000 --rescale_pixel_value=True --eval_every=5000 --mutation_rate=0.0 --wasserstein --reg_weight=0.0 --learning_rate {lr} --lambda_penalty {p} --wnorm {n} --random_seed {p}\n""")

# # MLE
# for p in [0]: #, 100, 1000]:
#     for lr in [.001, .0001]:
#         commands.append(f"""python -m genomics_ood.images_ood.train --exp=cifar --data_dir=../data --out_dir=genomics_ood/images_ood/cifar/mle_lr{lr}_p{p} --total_steps=600000 --rescale_pixel_value=True --eval_every=5000 --mutation_rate=0.0 --reg_weight=0.0 --learning_rate {lr} --random_seed {p}\n""")

# with open('jobsw_cifar.txt', 'w') as f:
#     f.writelines(commands)

# for lr in [.01, .001, .0001, .00001]:
#     for output in ['v1', 'v3']:
#         commands.append(f"""python -m genomics_ood.images_ood.train --exp=fashion --data_dir=../data --out_dir=genomics_ood/images_ood/fashionk_runs/mle_lr{lr}_p100/ --total_steps=100000 --rescale_pixel_value=False --eval_every=5000 --mutation_rate=0.0 --reg_weight=0.0 --learning_rate 0.001 --random_seed 100 --pixel_hist --grad_hist --dist kumaraswamy --output {output}\n""")

# one logistic
# commands.append(f"""python -m genomics_ood.images_ood.train --exp=fashion --data_dir=../data --out_dir=genomics_ood/images_ood/fashion/mle_lr{lr}_p100_grad/ --total_steps=100000 --rescale_pixel_value=False --eval_every=5000 --mutation_rate=0.0 --reg_weight=0.0 --learning_rate 0.001 --random_seed 100 --pixel_hist --grad_hist\n""")
# with open('jobsk.txt', 'w') as f:
#     f.writelines(commands)
for lr in [0.0001]:  #, 0.0001]:
    for v in ['v4', 'v5']:
        for n_mixtures in [1, 2]:
            for dist in ['logistic_transform', 'normal_transform']:
                if dist == 'logistic':
                    d = 'l'
                elif dist == 'logistic_transform':
                    d = 'lt'
                elif dist == 'normal_transform':
                    d = 'nt'
                commands.append(f"""python -m genomics_ood.images_ood.train --exp=cifar --data_dir=../data --out_dir=genomics_ood/images_ood/cifar2/mle_lr{lr}_m{n_mixtures}_{d}_{v}/ --total_steps=600000 --rescale_pixel_value=False --eval_every=5000 --mutation_rate=0.0 --reg_weight=0.0 --learning_rate {lr} --random_seed 100 --pixel_hist --grad_hist --dist {dist} --num_logistic_mix {n_mixtures} --output {v} --rescale_pixel_value=False\n""")

with open('jobst.txt', 'w') as f:
    f.writelines(commands)
