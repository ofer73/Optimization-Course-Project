import os
from datetime import datetime
from itertools import product
import sys

slurm = 'main.slurm'
cmd = '/home/ycarmon/users/maorkehati/anaconda3/envs/optp/bin/python ./src/main.py'

args_names = ["optim-method", "eta0", "alpha", "nesterov", "momentum", "weight-decay", "train-epochs", "batchsize", "eval-interval", "use-cuda", "dataset", "dataroot", "plot-lr", "tail-epochs"]

optim_method = ["SGD_Cosine_Start_Linear_Tail_Decay"]
eta0 = [float(10**i) for i in range(-5,1)]
eta0 = [0.01,0.1]
eta0 = [0.01]
#eta0 = [10**i for i in range(-5,-3)]
alpha = [None]
nesterov = [""] #[None] to disable
momentum = [0.9]
weight_decay = [0.0001]
train_epochs = [50]
#train_epochs = [1]
batchsize = [128]
eval_interval = [1]
use_cuda = [""]
dataset = ["FashionMNIST"]
dataroot = ["./data"]
plot_lr = [""]
tail_epochs = list(range(0, 32, 2))
tail_epochs = [12]
#tail_epochs = list(range(0, 5, 5))

def main():
    arg_values = [globals()[args_name.replace("-","_")] for args_name in args_names]
    args_list = list(product(*arg_values))

    args_list_print = [" ".join([f"{arg_name}:{arg_value}" for arg_name, arg_value in zip(args_names, args) if arg_value != None]) for args in args_list]
    args_list_print = "\n".join(args_list_print)
    print("using args list:\n"+args_list_print)
    
    if len(sys.argv) > 1:
        folder_name = sys.argv[1]
    else:
        folder_name = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    out_folder_root = f"outs/{folder_name}"
    os.makedirs(out_folder_root)
    
    with open(f'{out_folder_root}/args_iterate','w') as argsh:
        argsh.write("\n".join([arg_name.replace("-","_") for arg_name,arg_value in zip(args_names, arg_values) if len(arg_value)>1]))
    
    for args_values in args_list:
        print("args_values", args_values)
        folder_name = "_".join([f"{arg_name}-{arg_value if type(arg_value) != str else arg_value.replace('/','')}" for arg_name, arg_value in zip(args_names, args_values) if arg_value != None])
        out_folder = f"{out_folder_root}/{folder_name}"
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
        c = c[:cind]+ f'--job-name={folder_name}' + c[c.find("\n",cind):]
           
        
        with open(f'{folder_name}.slurm','w') as mainf:
            mainf.write(c)
            
        os.system(f'sbatch {folder_name}.slurm')

if __name__ == '__main__':
    main()