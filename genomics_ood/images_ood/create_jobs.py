commands = []

# Wasserstein
for p in [0]: #, 100, 1000]: #, 10000]:
    for lr in [.001, .0001]:  #[.1, .01, .001, .0001, .00001]:
        for n in [0, 1, 2]:  #[1, 2]:
            commands.append(f"""python -m genomics_ood.images_ood.train --exp=cifar --data_dir=../data --out_dir=genomics_ood/images_ood/cifar/w_lr{lr}_p{p}_n{n} --total_steps=600000 --rescale_pixel_value=True --eval_every=5000 --mutation_rate=0.0 --wasserstein --reg_weight=0.0 --learning_rate {lr} --lambda_penalty {p} --wnorm {n} --random_seed {p}\n""")

# MLE
for p in [0]: #, 100, 1000]:
    for lr in [.001, .0001]:
        commands.append(f"""python -m genomics_ood.images_ood.train --exp=cifar --data_dir=../data --out_dir=genomics_ood/images_ood/cifar/mle_lr{lr}_p{p} --total_steps=600000 --rescale_pixel_value=True --eval_every=5000 --mutation_rate=0.0 --reg_weight=0.0 --learning_rate {lr} --random_seed {p}\n""")

with open('jobsw_cifar.txt', 'w') as f:
    f.writelines(commands)
