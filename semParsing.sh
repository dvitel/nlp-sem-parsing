#!/bin/bash -l
#All options below are recommended
#SBATCH -o std_out
#SBATCH -e std_err
#SBATCH -D /data/dvitel/semParse
#SBATCH -p Quick # run on partition general
#SBATCH -w GPU44
#SBATCH --gpus=1 # 1 GPU
conda activate semParseEnv
srun python semParsing.py #run semantic parsing