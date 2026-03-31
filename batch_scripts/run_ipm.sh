#!/bin/bash
#SBATCH -J run_ipm
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -t 08:00:00
#SBATCH -o logs/run_ipm_%j.out

source /users/kerdos/miniconda3/etc/profile.d/conda.sh

echo "trying to run script"

PYTHON_FILE=../run.py
DATAPATH=../instance_data/

OUTDIR=/users/kerdos/out.txt

conda run -n ipmgnn python -u $PYTHON_FILE --datapath $DATAPATH \
    --ipm_alpha 0.76 \
    --weight_decay 4.4e-7 \
    --batchsize 512 \
    --hidden 180 \
    --num_pred_layers 3 \
    --num_mlp_layers 4 \
    --share_lin_weight false \
    --conv_sequence cov \
    --loss_weight_x 1. \
    --loss_weight_obj 0.33 \
    --loss_weight_cons 2.2 \
    --runs 3 \
    --lappe 0 \
    --conv gcnconv > $OUTDIR 2>&1

echo "done!"
