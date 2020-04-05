#!/bin/bash

#SBATCH --job-name DownloadImg
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4g
#SBATCH --time=01:00:00
#SBATCH --account=si699w20_cbudak_class
#SBATCH --partition=standard
#SBATCH --mail-type=END,FAIL

module purge
module load python2.7-anaconda/2019.03
module list

#python get_img_list.py
python tmp.py
