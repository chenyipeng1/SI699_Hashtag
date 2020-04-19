#!/bin/bash

#SBATCH --job-name=pytorch_combined_model
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=1g
#SBATCH --time=40:00:00
#SBATCH --account=si699w20_cbudak_class
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

source start.sh
python model/text_train.py
