#!/bin/bash

#SBATCH --job-name=pytorch_combined_model
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4g
#SBATCH --time=10:00:00
#SBATCH --account=si699w20_cbudak_class
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module purge
module load python2.7-anaconda/2019.03
module list

#python get_img_list.py
python tmp.py
