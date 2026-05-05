#!/bin/bash
#SBATCH --job-name=sft-qwen35-08b-full
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --time=05:00:00

echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job name: $SLURM_JOB_NAME"
echo "Running on node: $(hostname)"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "============================================================"

module purge
module load anaconda3/2025.6
# conda activate torch-env
# python myscript.py

cd /scratch/network/jw5134/COS598B-project/
source .venv/bin/activate

mkdir -p logs
mkdir -p results



echo "============================================================"
echo "Starting full-parameter SFT"
echo "============================================================"

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

python sft_qwen35_08b_full.py \
  --model_id "Qwen/Qwen3.5-0.8B-Base" \
  --sat_train_path "dataset/sft_by_label/sat/train.jsonl" \
  --unsat_train_path "dataset/sft_by_label/unsat/train.jsonl" \
  --output_dir "results/sft_qwen35_08b_base_sat_unsat_full" \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --max_length 4096 \
  --logging_steps 1 \
  --save_steps 10 \
  --save_total_limit 3

echo "============================================================"
echo "End time: $(date)"
echo "============================================================"