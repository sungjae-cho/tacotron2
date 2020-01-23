import time
import torch
import sys
import subprocess
import os

argslist = list(sys.argv)[1:]
for args in argslist:
    if "--visible_gpus=" in args:
        v_gpus = args[len("--visible_gpus="):]
        os.environ["CUDA_VISIBLE_DEVICES"] = v_gpus
        print(os.environ["CUDA_VISIBLE_DEVICES"])
    if "--n_gpus=" in args:
        num_gpus = int(args[len("--n_gpus="):])
        print(num_gpus)

#num_gpus = torch.cuda.device_count()
#argslist.append('--n_gpus={}'.format(num_gpus))

workers = []
job_id = time.strftime("%Y_%m_%d-%H%M%S")
argslist.append("--group_name=group_{}".format(job_id))

for i in range(num_gpus):
    argslist.append('--rank={}'.format(i))
    stdout = None if i == 0 else open("logs/{}_GPU_{}.log".format(job_id, i),
                                      "w")
    print(argslist)
    p = subprocess.Popen([str(sys.executable)]+argslist, stdout=stdout)
    workers.append(p)
    argslist = argslist[:-1]

for p in workers:
    p.wait()
