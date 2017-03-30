#!/bin/bash -l
#SBATCH -N 1
#SBATCH -t 420
#SBATCH -p regular
#SBATCH --qos=premium
#SBATCH -A dasrepo

bash ./get_accuracy.sh --path_to_caffemodel /global/cscratch1/sd/imit/gbruns/climate/BEST_CLIMATE_LONG/snapshot/BEST_CLIMATE_LONG/lr_0.0001_mu_0.97_step_20000_step-fac_1.0_clip_1000000000.0_decay_0.0005_rand_1_iter_1200.caffemodel --path_to_caffe_prototxt /project/projectdirs/dasrepo/gordon_bell/deep_learning/networks/climate/2d_semi_sup/base_prototxt_files/deploy_vanilla_train.prototxt $@
