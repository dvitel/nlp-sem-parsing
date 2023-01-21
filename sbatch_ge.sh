#!/bin/bash -l
sbatch -J "ge-${1//\./_}" -o "ge-${1//\./_}.out" -e "ge-${1//\./_}.out" ge.sh
