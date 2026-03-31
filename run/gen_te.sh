#!/bin/bash
#SBATCH -J te_gen
#SBATCH -p batch
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -o logs/te_gen_%j.out

source ~/.bashrc
conda activate ipmgnn

cd /users/rkapoor8/CS2680/recreateTelgen

python generate_te_instances.py \
  --output_dir datasets/te_small \
  --num_instances 200 \
  --instances_per_file 200 \
  --seed 0 \
  --n_nodes_min 8 \
  --n_nodes_max 14 \
  --edge_prob 0.25 \
  --num_demands_min 6 \
  --num_demands_max 16 \
  --demand_min 5 \
  --demand_max 40 \
  --cap_min 30 \
  --cap_max 120 \
  --k_paths 3