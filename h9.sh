#!/bin/bash -l
#All options below are recommended
#SBATCH --job-name=semp-h9
#SBATCH -o h9.out
#SBATCH -e h9.err
#SBATCH -D /data/dvitel/semParse
#SBATCH -p Quick # run on partition general
#SBATCH --gpus=1 # 1 GPU
conda activate semParse2
srun /home/d/dvitel/semp/start.sh '/home/d/dvitel/semp/h9'
