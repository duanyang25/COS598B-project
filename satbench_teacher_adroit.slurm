#!/bin/bash
#SBATCH --job-name=satbench_teacher
#SBATCH --output=logs/satbench_teacher_%A_%a.out
#SBATCH --error=logs/satbench_teacher_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --time=24:00:00
#SBATCH --array=0-20

# Slurm array script for generating SATBench teacher traces on Princeton Adroit.
#
# Submit from the directory containing:
#   - generate_satbench_teacher_traces_adroit.py
#   - satbench_with_certificates_full.jsonl
#
# Each array task processes 100 rows:
#   task 0  -> rows [0, 100)
#   task 1  -> rows [100, 200)
#   ...
#   task 20 -> rows [2000, 2100)
#
# This requests exactly one GPU per task. Do not manually set GPU groups.
# Slurm will set CUDA_VISIBLE_DEVICES for the GPU assigned to each job.
#
# --mem=64G is CPU/system RAM, not GPU memory. If model loading fails due to
# host-memory pressure, change it to --mem=96G or --mem=128G.

set -euo pipefail

mkdir -p logs teacher_traces

# Activate your Python environment. Edit this section for your setup.
# Example with conda:
#   source ~/miniconda3/etc/profile.d/conda.sh
#   conda activate satbench
# Example with venv:
#   source ~/satbench-venv/bin/activate

# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate satbench

# Offline/cache settings. Keep these if the model is already downloaded.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# If your model cache is in a non-default location, uncomment and edit:
# export HF_HOME=/path/to/your/huggingface/cache

python generate_satbench_teacher_traces_adroit.py \
  --dataset satbench_with_certificates_full.jsonl \
  --output_dir teacher_traces \
  --model_id Qwen/Qwen3.6-35B-A3B \
  --local_files_only \
  --dtype bfloat16 \
  --device_map auto \
  --shard_idx "${SLURM_ARRAY_TASK_ID}" \
  --shard_size 100 \
  --max_new_tokens 8192 \
  --temperature 1.0 \
  --top_p 0.95 \
  --top_k 20 \
  --repetition_penalty 1.0 \
  --log_every 5
