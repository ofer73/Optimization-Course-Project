import os
from datetime import datetime
from itertools import product
import shutil
import sys
import time
import random
import numpy as np
import math

slurm = 'main.slurm'
cmd = '/home/ycarmon/users/maorkehati/anaconda3/envs/optp/bin/python ./src/main.py'
sleep_time = 20

args_names = ["optim-method", "eta0", "alpha", "nesterov", "momentum", "weight-decay", "train-epochs", "batchsize", "eval-interval", "use-cuda", "dataset", "dataroot", "plot-lr", "tail-epochs","validation"]

#optim_method = ["SGD_Cosine_Start_Linear_Tail_Decay"]
optim_method = ["SGD_Exp_Start_Cosine_Tail_Decay"]
eta0 = [0.01]
alpha = [0.005,0.01,0.025,0.05,0.075,0.1] #this represents the ratios not the alpha values!
nesterov = [""] #[None] to disable
momentum = [0.9]
weight_decay = [0.0001]
batchsize = [128]
eval_interval = [1]
use_cuda = [""]
dataset = ["FashionMNIST"] #FashionMNIST CIFAR10 CIFAR100
dataroot = ["./data"]
plot_lr = [""]
validation = [""]
tail_epochs = [0]#list(range(0, 55, 5))
TIMES = 30

if len(dataset) == 1:
    if dataset[0] == "FashionMNIST":
        train_epochs = [50]
        
    elif dataset[0] == "CIFAR10":
        train_epochs = [164]

    elif dataset[0] == "CIFAR100":
        train_epochs = [50]

train_lens = {'FashionMNIST':60000, 'CIFAR10':50000, 'CIFAR100':50000}

run_crashed = []#[[('0.1', '0'), 0], [('0.1', '0'), 4], [('0.1', '10'), 0], [('0.1', '25'), 3], [('0.1', '30'), 3], [('0.1', '35'), 8], [('0.1', '45'), 0], [('0.1', '45'), 1]]

def main():
    is_run_crashed = 'run_crashed' in globals() and run_crashed
    
    if is_run_crashed:
        assert len(sys.argv) > 1
        folder_name = sys.argv[1]
        out_folder_root = f"outs/{folder_name}"
        assert os.path.exists(out_folder_root)
    else:
        if len(sys.argv) > 1:
            folder_name = sys.argv[1]
        else:
            folder_name = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        out_folder_root = f"outs/{folder_name}"
        if os.path.exists(out_folder_root):
            if len(sys.argv)>2 and sys.argv[2] == 'ow':
                shutil.rmtree(out_folder_root)
            else:
                print(f'folder name exists. Run "{sys.argv[0]} <foldername> ow" to overwrite')
                return
        os.makedirs(out_folder_root)
    
    arg_values = [globals()[args_name.replace("-","_")] for args_name in args_names]
    args_iterate = [arg_name for arg_name,arg_value in zip(args_names, arg_values) if len(arg_value)>1]
    if 'alpha' in args_iterate:
        args_iterate[args_iterate.index('alpha')] = 'alpha_name'

    if is_run_crashed:
        assert all([len(i[0]) == len(args_iterate) for i in run_crashed])
        args_list = []
        for run_crashed_values in run_crashed:
            run_arg_values = list(map(lambda x: x[0], arg_values))
            argsh_indices = [args_names.index(arg_name) for arg_name in args_iterate]
            for argsh_index, argsh_value in zip(argsh_indices,run_crashed_values[0]):
                run_arg_values[argsh_index] = argsh_value
                
            args_list.append((run_arg_values,run_crashed_values[1]))
        args_iterate = [arg_name.replace("-","_") for arg_name in args_iterate]
        
    else:
        args_list_pre = [list(i) for i in product(*arg_values)]#list(map(list, product(*arg_values)))
        args_list = []
        for args_list_row in args_list_pre:
            for i in range(TIMES):
                args_list.append((args_list_row.copy(), i))
        
        args_iterate = [arg_name.replace("-","_") for arg_name in args_iterate]
        with open(f'{out_folder_root}/args_iterate','w') as argsh:
            argsh.write(f"TIMES={TIMES}\n")
            argsh.write("\n".join(args_iterate))
        
        #turn alpha ratios into alphas
        dataset_index = args_names.index("dataset")
        alphas_index = args_names.index("alpha")
        epochs_index = args_names.index("train-epochs")
        bs_index = args_names.index("batchsize")
        if len(alpha)>0 and alpha[0]:
            args_names.append('alpha-name')
            for ai, (alist,alist_iter_num) in enumerate(args_list):
                if not alist[alphas_index]:
                    continue
                    
                alist[alphas_index] = float(alist[alphas_index])
                alist.append(alist[alphas_index])
                alist[alphas_index] **= (1/(math.ceil(train_lens[alist[dataset_index]]/alist[bs_index]) * float(alist[epochs_index])))
                args_list[ai] = (alist, alist_iter_num)
    
    args_list_print = [" ".join([f"{arg_name}:{arg_value}" for arg_name, arg_value in zip(args_names, args[0]) if arg_value != None]) + f"\t#{args[1]}" for args in args_list]
    args_list_print = "\n".join(args_list_print)
    print("using args list:\n"+args_list_print)
    
    slurms = []
    jobs_count = 0
    for args_values, iter_num in args_list:
        print("args_values", args_values)
        folder_name_base = "_".join([f"{arg_name.replace('-','_')}-{arg_value if type(arg_value) != str else arg_value.replace('/','')}" for arg_name, arg_value in zip(args_names, args_values) if arg_value != None and arg_name.replace('-','_') in args_iterate])
        ext = ''
        if TIMES > 1:
            ext = f'_{iter_num}'               
            
        folder_name = f'{folder_name_base}{ext}'
        out_folder = f"{out_folder_root}/{folder_name}"
        if is_run_crashed:
            shutil.rmtree(out_folder)
            
        os.makedirs(out_folder)
        with open(slurm,'r') as mainf:
            c = mainf.read()
            
        cind = c.find(cmd)
        c = c[:cind]+ cmd + f" --log-folder {out_folder} " + " ".join([f"--{arg_name} {arg_value}" for arg_name, arg_value in zip(args_names, args_values) if arg_value != None]) + c[c.find("\n",cind):]
        cind = c.find('--output')
        c = c[:cind]+ f'--output={out_folder}/out.out' + c[c.find("\n",cind):]
        cind = c.find('--error')
        c = c[:cind]+ f'--error={out_folder}/err.err' + c[c.find("\n",cind):]
        
        cind = c.find('--job-name=')
        c = c[:cind]+ f'--job-name=optim-{folder_name}' + c[c.find("\n",cind):]
        
        rand_slurm_name = f'{time.time()}_{random.random()}'
        with open(f'{rand_slurm_name}.slurm','w') as mainf:
            mainf.write(c)
        
        print(folder_name)
        os.system(f'sbatch {rand_slurm_name}.slurm')
        slurms.append(rand_slurm_name)
        jobs_count += 1
    
    print(f'started {jobs_count} jobs in {out_folder_root}')
    print(f'sleeping {sleep_time}s...')
    time.sleep(sleep_time)
    for slurm_name in slurms:
        os.remove(f'{slurm_name}.slurm')

if __name__ == '__main__':
    main()