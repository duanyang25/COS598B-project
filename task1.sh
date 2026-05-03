#!/bin/bash
#SBATCH --job-name=task1-teacher            # create a short name for your job
#SBATCH --nodes=1                   # node count
#SBATCH --ntasks=1                  # total number of tasks across all nodes
#SBATCH --cpus-per-task=8           # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=64G            # total memory
#SBATCH --gres=gpu:1                # number of gpus per node
#SBATCH --constraint=gpu80
#SBATCH --time=00:30:00             # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin,end,fail  # receive email notifications
#SBATCH --mail-user=yd1202@princeton.edu

module purge
module load anaconda3/2025.6
# conda activate torch-env
# python myscript.py

cd /scratch/network/yd1202/COS598B-project/

source .venv/bin/activate

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# vllm serve Qwen/Qwen3.6-35B-A3B --port 11148 --tensor-parallel-size 1 --max-model-len 262144 --reasoning-parser qwen3 --language-model-only
