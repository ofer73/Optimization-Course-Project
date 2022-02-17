#!/bin/sh
#SBATCH --output=out.out
#SBATCH --error=err.err
#SBATCH --partition=studentkillable
#SBATCH --job-name=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
# SBATCH --cpus-per-task=20
# SBATCH --mem=128G
#SBATCH --time=5-00:00:00
# SBATCH --constraint="geforce_rtx_3090"
# SBATCH --nodelist=n-302

/home/ycarmon/users/maorkehati/anaconda3/envs/optp/bin/python ./src/main.py --optim-method SGD --eta0 0.007 --nesterov --momentum 0.9 --weight-decay 0.0001 --train-epochs 50 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/FashionMNIST --dataset FashionMNIST --dataroot ./data
