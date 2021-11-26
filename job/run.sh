# Note: all paths referenced here are relative to the Docker container.
#
# Add the Nvidia drivers to the path
export PATH="/usr/local/nvidia/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/tools/cuda/lib64
# Tools config for CUDA, Anaconda installed in the common /tools directory
source /tools/config.sh
# Activate your environment
source activate torch37
# source activate /scratch/scratch6/akansh12/env
# Change to the directory in which your code is present
cd /storage/home/akansh12/test/kube-test/job
# Run the code. The -u option is used here to use unbuffered writes
# so that output is piped to the file as and when it is produced.
# Here, the code is a simple Pytorch script to check if we are using the GPU.
# python -u firsttimecluster.py &> out
python -u local_labels.py &> out

