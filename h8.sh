#!/bin/bash -l
#All options below are recommended
#SBATCH --job-name=semp-h8
#SBATCH -o h8.out
#SBATCH -e h8.err
#SBATCH -D /data/dvitel/semParse
#SBATCH -p Quick # run on partition general
#SBATCH --gpus=1 # 1 GPU
conda activate semParse2
srun /home/d/dvitel/semp/start.sh '/home/d/dvitel/semp/h8'
