#!/bin/bash
#SBATCH -n 4
#SBATCH --mem=16G
#SBATCH -t 1:00:00

source /users/kerdos/miniconda3/etc/profile.d/conda.sh

echo "trying to run python"

conda run -n ipmgnn python -u test_torch.py > /users/kerdos/out.txt

echo "done!"
