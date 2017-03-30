export HDF5_DISABLE_VERSION_CHECK=1
module load master_module
python -u get_accuracy.py $@
