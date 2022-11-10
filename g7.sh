#!/bin/bash -l
#All options below are recommended
#SBATCH --job-name=semp-g7
#SBATCH -o g7.out
#SBATCH -e g7.err
#SBATCH -D /data/dvitel/semParse
#SBATCH -p Quick # run on partition general
#SBATCH --gpus=1 # 1 GPU
#SBATCH --exclude GPU41
conda activate semParse2
srun /home/d/dvitel/semp/start.sh '/home/d/dvitel/semp/g7'
