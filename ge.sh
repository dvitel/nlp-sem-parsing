#!/bin/bash -l
#All options below are recommended
#SBATCH -D /data/dvitel/semParse
#SBATCH -p Quick # run on partition general
#SBATCH --gpus=1 # 1 GPU
#SBATCH -w GPU45
conda activate semParse2
srun -J "ge-${1//\./_}" -o "ge-${1//\./_}.out" -e "ge-${1//\./_}.err" python3 /home/d/dvitel/semp/ge.py $1 $2