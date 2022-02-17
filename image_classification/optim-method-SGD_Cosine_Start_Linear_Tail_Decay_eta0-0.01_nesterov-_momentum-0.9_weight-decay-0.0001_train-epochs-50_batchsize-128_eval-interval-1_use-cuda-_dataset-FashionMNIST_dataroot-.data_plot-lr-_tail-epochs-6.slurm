#!/bin/sh
#SBATCH --output=outs/2022_02_17_01_35_04/optim-method-SGD_Cosine_Start_Linear_Tail_Decay_eta0-0.01_nesterov-_momentum-0.9_weight-decay-0.0001_train-epochs-50_batchsize-128_eval-interval-1_use-cuda-_dataset-FashionMNIST_dataroot-.data_plot-lr-_tail-epochs-6/out.out
#SBATCH --error=outs/2022_02_17_01_35_04/optim-method-SGD_Cosine_Start_Linear_Tail_Decay_eta0-0.01_nesterov-_momentum-0.9_weight-decay-0.0001_train-epochs-50_batchsize-128_eval-interval-1_use-cuda-_dataset-FashionMNIST_dataroot-.data_plot-lr-_tail-epochs-6/err.err
#SBATCH --partition=studentkillable
#SBATCH --job-name=optim-method-SGD_Cosine_Start_Linear_Tail_Decay_eta0-0.01_nesterov-_momentum-0.9_weight-decay-0.0001_train-epochs-50_batchsize-128_eval-interval-1_use-cuda-_dataset-FashionMNIST_dataroot-.data_plot-lr-_tail-epochs-6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
# SBATCH --cpus-per-task=20
# SBATCH --mem=128G
#SBATCH --time=5-00:00:00
# SBATCH --constraint="geforce_rtx_3090"
# SBATCH --nodelist=n-302

/home/ycarmon/users/maorkehati/anaconda3/envs/optp/bin/python ./src/main.py --log-folder outs/2022_02_17_01_35_04/optim-method-SGD_Cosine_Start_Linear_Tail_Decay_eta0-0.01_nesterov-_momentum-0.9_weight-decay-0.0001_train-epochs-50_batchsize-128_eval-interval-1_use-cuda-_dataset-FashionMNIST_dataroot-.data_plot-lr-_tail-epochs-6 --optim-method SGD_Cosine_Start_Linear_Tail_Decay --eta0 0.01 --nesterov  --momentum 0.9 --weight-decay 0.0001 --train-epochs 50 --batchsize 128 --eval-interval 1 --use-cuda  --dataset FashionMNIST --dataroot ./data --plot-lr  --tail-epochs 6
