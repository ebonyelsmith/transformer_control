#!/bin/bash
#SBATCH --job-name=multi_gpu_job
#SBATCH --gres=gpu:3          # 3 GPUs
#SBATCH --cpus-per-task=96    # All cpus
#SBATCH --mem=600G            # ~600GB total
#SBATCH --time=72:00:00       # Up to 72 hours

# Load environment
# source ~/miniconda3/etc/profile.d/conda.sh
# source /opt/miniconda/condabin/conda.sh
source /opt/miniconda/etc/profile.d/conda.sh

conda activate icl_ebonye_pendulum

which python
which torchrun

# Multiple GPU setup
# python -m torch.distributed.launch --nproc_per_node=2 trainSequential_ebonye_acrobot.py --config conf/_XandYtest_acrobot_ebonye.yaml
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
# torchrun --nnodes=1 --nproc_per_node=2 trainSequential_ebonye_acrobot.py --config conf/_XandYtest_acrobot_ebonye.yaml
# torchrun --nnodes=1 --nproc_per_node=2 trainSequential_ebonye_cartpole.py --config conf/_XandYtest_cartpole_ebonye.yaml
torchrun --nnodes=1 --nproc_per_node=2 trainSequential_ebonye_cartpole_zerodyn.py --config conf/_XandYtest_cartpole_ebonye.yaml