#!/bin/bash -l
#All options below are recommended
#SBATCH --job-name=semp-g1
#SBATCH -o g1.out
#SBATCH -e g1.err
#SBATCH -D /data/dvitel/semParse
#SBATCH -p Quick # run on partition general
#SBATCH --gpus=1 # 1 GPU
#SBATCH --exclude GPU41
conda activate semParse2
srun /home/d/dvitel/semp/start.sh '/home/d/dvitel/semp/g1'
