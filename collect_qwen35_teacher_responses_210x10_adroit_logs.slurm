#!/bin/bash
#SBATCH --job-name=qwen35-2b-teacher
#SBATCH --output=logs/qwen35_teacher_%A_%a.out
#SBATCH --error=logs/qwen35_teacher_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --time=00:30:00
#SBATCH --array=88-209%4
## Optional email notifications. Replace <YourNetID> before uncommenting.
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=yd1202@princeton.edu

# This array has 210 jobs.
# Each job processes 10 rows from dataset/satbench_teacher_prompts.jsonl:
#   task 0   -> rows 0-9
#   task 1   -> rows 10-19
#   ...
#   task 209 -> rows 2090-2099
# The %20 cap means at most 20 array tasks run at the same time.
# Increase/decrease %20 based on queue pressure and available GPUs.

module purge
module load anaconda3/2025.6
# conda activate torch-env
# python myscript.py

cd /scratch/network/yd1202/COS598B-project/
source .venv/bin/activate


set -euo pipefail

# The logs/ directory must exist before sbatch because Slurm opens stdout/stderr paths before this script runs.
# You said you already created it; keep this mkdir as a harmless extra safety check for non-Slurm-created paths.
mkdir -p logs teacher_responses_qwen35_2b/records teacher_responses_qwen35_2b/logs

echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "Host=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"


# Offline mode is useful after the model is already downloaded/cached.
# Comment these out if you want Hugging Face to download missing files.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Keep CPU thread usage modest. Each task owns one GPU and four CPU cores.
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROMPTS="dataset/satbench_teacher_prompts.jsonl"
OUTPUT_DIR="teacher_responses_qwen35_2b"
MODEL_ID="Qwen/Qwen3.5-2B"
SHARD_SIZE=10

# Qwen3.5-2B is small enough that an 80GB A100 constraint is not needed.
# dtype=auto is safest across mixed GPU types. If you know the node is V100, float16 is also fine.
DTYPE="auto"
BATCH_SIZE=1
MAX_NEW_TOKENS=8192

python collect_qwen35_teacher_responses_resumable_logs.py \
  --mode generate \
  --prompts "${PROMPTS}" \
  --output_dir "${OUTPUT_DIR}" \
  --model_id "${MODEL_ID}" \
  --local_files_only \
  --dtype "${DTYPE}" \
  --device_map auto \
  --shard_idx "${SLURM_ARRAY_TASK_ID}" \
  --shard_size "${SHARD_SIZE}" \
  --expected_rows 2100 \
  --batch_size "${BATCH_SIZE}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --do_sample \
  --temperature 0.7 \
  --top_p 0.95 \
  --top_k 20 \
  --repetition_penalty 1.05 \
  --log_every 1
