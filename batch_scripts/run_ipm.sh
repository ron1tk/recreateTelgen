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

PYTHON_FILE=

conda run -n ipmgnn python -u $(PYTHON_FILE) > /users/kerdos/out.txt

echo "done!"
