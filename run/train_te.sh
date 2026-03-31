#!/bin/bash
#SBATCH -J te_train
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -t 08:00:00
#SBATCH -o logs/te_train_%j.out

source ~/.bashrc
conda activate ipmgnn

cd /users/rkapoor8/CS2680/recreateTelgen

python run.py \
  --datapath /users/rkapoor8/CS2680/recreateTelgen/datasets/te_small \
  --model_variant telgen \
  --conv gcnconv \
  --bipartite false \
  --batchsize 32 \
  --epoch 300 \
  --patience 200 \
  --lr 3e-4 \
  --ipm_steps 8 \
  --ipm_alpha 0.8 \
  --upper 1.0 \
  --loss primal+objgap+constraint \
  --loss_weight_x 1.0 \
  --loss_weight_obj 1.3 \
  --loss_weight_cons 4.6 \
  --hidden 180 \
  --num_conv_layers 8 \
  --num_inner_layers 2 \
  --num_pred_layers 4 \
  --num_mlp_layers 4 \
  --dropout 0.0 \
  --use_norm true \
  --use_res false \
  --conv_sequence cov \
  --ckpt true