#!/bin/bash -l
#All options below are recommended
#SBATCH --job-name=ge-${1//\./_}
#SBATCH -o ge-${1//\./_}.out
#SBATCH -e ge-${1//\./_}.err
#SBATCH -D /data/dvitel/semParse
#SBATCH -p Quick # run on partition general
#SBATCH --gpus=1 # 1 GPU
#SBATCH -w GPU45
conda activate semParse2
srun python3 /home/d/dvitel/semp/ge.py $1 $2
