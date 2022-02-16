import os
from datetime import datetime
from itertools import product

slurm = 'main.slurm'
cmd = '/home/ycarmon/users/maorkehati/anaconda3/envs/optp/bin/python ./src/main.py'

args_names = ["optim-method", "eta0", "alpha", "nesterov", "momentum", "weight-decay", "train-epochs", "batchsize", "eval-interval", "use-cuda", "log-folder", "dataset", "dataroot"]

optim_method = ["SGD_Cosine_Start_Linear_Tail_Decay"]
eta0 = [10**i for i in range(-5,1)]
alpha = [None]
nesterov = [""]
momentum = [0.9]
weight_decay = [0.0001]
train_epochs = [50]
batchsize = [128]
eval_interval = [1]
use_cuda = [""]
log_folder = [".logs/FashionMNIST"]
dataset = ["FashionMNIST"]
dataroot = ["./data"]

args_list = list(product(*[globals()[i.replace("-","_")] for i in args_names]))

args_list_print = [" ".join([f"{arg_name}:{arg_value}" for arg_name, arg_value in zip(args_names, args) if arg_value != None]) for args in args_list]
args_list_print = "\n".join(args_list_print)
print("using args list:\n"+args_list_print)

def main():
    out_folder_root = f"outs/{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
    os.makedirs(out_folder_root)
    for args_values in args_list:
        print("args_values", args_values)
        folder_name = "_".join([f"{arg_name}-{arg_value if type(arg_value) != str else arg_value.replace('/','')}" for arg_name, arg_value in zip(args_names, args_values)])
        out_folder = f"{out_folder_root}/{folder_name}"
        os.makedirs(out_folder)
        with open(slurm,'r') as mainf:
            c = mainf.read()
            
        cind = c.find(cmd)
        c = c[:cind]+ cmd + " " + " ".join([f"--{arg_name} {arg_value}" for arg_name, arg_value in zip(args_names, args_values) if arg_value != None]) + c[c.find("\n",cind):]
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