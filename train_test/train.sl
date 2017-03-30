#!/bin/bash -l
#SBATCH -N 1
#SBATCH -C haswell
#SBATCH -t 16:00:00
#SBATCH -p regular
#SBATCH --qos=premium
#SBATCH -e slurm_outputs/slurm-%A.out
#SBATCH -o slurm_outputs/slurm-%A.out
module load /global/homes/r/racah/projects/caffe-netcdf/caffe/modules/master_module
python make_network.py $@
