#!/bin/bash -l
#All options below are recommended
#SBATCH --job-name=semp-h0-1
#SBATCH -o h0-1.out
#SBATCH -e h0-1.err
#SBATCH -D /data/dvitel/semParse
#SBATCH -p Quick # run on partition general
#SBATCH --gpus=1 # 1 GPU
#SBATCH --exclude GPU41
conda activate semParse2
srun /home/d/dvitel/semp/start.sh '/home/d/dvitel/semp/h0-1'
