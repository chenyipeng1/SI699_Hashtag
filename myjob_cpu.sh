#!/bin/bash

#SBATCH --job-name=feiyi_cpu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2g
#SBATCH --time=10:00:00
#SBATCH --account=si699w20_cbudak_class

source start.sh
python model/train.py
