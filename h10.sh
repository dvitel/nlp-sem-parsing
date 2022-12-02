#!/bin/bash -l
#All options below are recommended
#SBATCH --job-name=semp-h10
#SBATCH -o h10.out
#SBATCH -e h10.err
#SBATCH -D /data/dvitel/semParse
#SBATCH -p Quick # run on partition general
#SBATCH --gpus=1 # 1 GPU
conda activate semParse2
srun /home/d/dvitel/semp/start.sh '/home/d/dvitel/semp/h10'
