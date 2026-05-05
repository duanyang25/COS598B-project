#!/bin/bash
#SBATCH --job-name=qwen35-08b-base-eval
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --time=00:05:00

echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Start time: $(date)"

module purge
module load anaconda3/2025.6
# conda activate torch-env
# python myscript.py

cd /scratch/network/jw5134/COS598B-project/
source .venv/bin/activate

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

python baseline_qwen35_08b_eval.py \
  --model_id "Qwen/Qwen3.5-0.8B-Base" \
  --sat_test_path "dataset/sft_by_label/sat/test.jsonl" \
  --unsat_test_path "dataset/sft_by_label/unsat/test.jsonl" \
  --output_dir "results/baseline_qwen35_08b_base_results" \
  --max_new_tokens 8192 \
  --stream 

echo "End time: $(date)"