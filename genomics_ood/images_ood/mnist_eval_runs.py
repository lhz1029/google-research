with open('jobsw.txt') as f:
    lines = f.readlines()

commands = [l.split(' --total_steps')[0] + "/expfashion/rescaleFalse --ckpt_step=-1 --repeat_id=0\n"
    for l in lines]
commands = [c.replace('train', 'eval_wasserstein') for c in commands]
with open('jobsw_eval.txt', 'w') as f:
    f.writelines(commands)