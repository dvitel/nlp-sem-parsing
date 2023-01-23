#!/bin/bash -l
sbatch -J "ge-$2-${1//\./_}" -o "ge-$2-${1//\./_}.out" -e "ge-$2-${1//\./_}.out" ge.sh "$@"