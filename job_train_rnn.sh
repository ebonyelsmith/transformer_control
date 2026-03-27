#!/bin/bash
source /opt/miniconda/etc/profile.d/conda.sh
conda activate icl_ebonye_pendulum_new
which python
which torchrun
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
torchrun --nnodes=1 --nproc_per_node=1 --master_port=29588 train_rnn_baseline.py --config conf/rnn_train_config.yaml