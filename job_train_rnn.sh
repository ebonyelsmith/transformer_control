#!/bin/bash
#SBATCH --job-name=multi_gpu_job
#SBATCH --gres=gpu:2          # 3 GPUs
#SBATCH --cpus-per-task=96    # All cpus
#SBATCH --mem=400G            # ~400GB total
#SBATCH --time=72:00:00       # Up to 72 hours

source /opt/miniconda/etc/profile.d/conda.sh
conda activate icl_ebonye_pendulum
which python
which torchrun
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
torchrun --nnodes=1 --nproc_per_node=1 --master_port=29592 train_rnn_baseline_cartpole.py --config conf/rnn_train_config_cartpole.yaml