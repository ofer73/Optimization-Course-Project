#!/bin/sh
#SBATCH --output=outs/2022_02_16_17_22_55/optim-method-SGD_Cosine_Start_Linear_Tail_Decay_eta0-0.1_alpha-None_nesterov-_momentum-0.9_weight-decay-0.0001_train-epochs-50_batchsize-128_eval-interval-1_use-cuda-_dataset-FashionMNIST_dataroot-.data_plot-lr-/out.out
#SBATCH --error=outs/2022_02_16_17_22_55/optim-method-SGD_Cosine_Start_Linear_Tail_Decay_eta0-0.1_alpha-None_nesterov-_momentum-0.9_weight-decay-0.0001_train-epochs-50_batchsize-128_eval-interval-1_use-cuda-_dataset-FashionMNIST_dataroot-.data_plot-lr-/err.err
#SBATCH --partition=studentkillable
#SBATCH --job-name=optim-method-SGD_Cosine_Start_Linear_Tail_Decay_eta0-0.1_alpha-None_nesterov-_momentum-0.9_weight-decay-0.0001_train-epochs-50_batchsize-128_eval-interval-1_use-cuda-_dataset-FashionMNIST_dataroot-.data_plot-lr-
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
# SBATCH --cpus-per-task=20
# SBATCH --mem=128G
#SBATCH --time=5-00:00:00
# SBATCH --constraint="geforce_rtx_3090"
# SBATCH --nodelist=n-302

/home/ycarmon/users/maorkehati/anaconda3/envs/optp/bin/python ./src/main.py --log-folder outs/2022_02_16_17_22_55/optim-method-SGD_Cosine_Start_Linear_Tail_Decay_eta0-0.1_alpha-None_nesterov-_momentum-0.9_weight-decay-0.0001_train-epochs-50_batchsize-128_eval-interval-1_use-cuda-_dataset-FashionMNIST_dataroot-.data_plot-lr- --optim-method SGD_Cosine_Start_Linear_Tail_Decay --eta0 0.1 --nesterov  --momentum 0.9 --weight-decay 0.0001 --train-epochs 50 --batchsize 128 --eval-interval 1 --use-cuda  --dataset FashionMNIST --dataroot ./data --plot-lr 
