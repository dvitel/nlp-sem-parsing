#!/bin/bash -l
#All options below are recommended
#SBATCH --job-name=ge1
#SBATCH -o ge1.out
#SBATCH -e ge1.err
#SBATCH -D /data/dvitel/semParse
#SBATCH -p Quick # run on partition general
#SBATCH --gpus=1 # 1 GPU
conda activate semParse2
srun /home/d/dvitel/semp/start.sh '/home/d/dvitel/semp/ge1'
