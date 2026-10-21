#!/bin/bash
# sbatch --export=MODEL=medgemma job_vlm.sh    (runs both datasets, all splits, all actions)
#SBATCH --job-name=cvlm
#SBATCH --partition=GPU-A100
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-12:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
set -euo pipefail
export LC_ALL=C PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
mkdir -p logs results
PY=~/miniconda3/envs/countervlm/bin/python
echo "host $(hostname) job ${SLURM_JOB_ID:-none} model ${MODEL} started $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
for SPLIT in ${SPLITS:-test validation train}; do
  for DS in ${DATASETS:-vqa_rad slake}; do
    $PY -u scripts/run_vlm.py --model "$MODEL" --dataset "$DS" --splits "$SPLIT" --batch_size "${BS:-16}"
  done
done
echo "finished $(date)"
