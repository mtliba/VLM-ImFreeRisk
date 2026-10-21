#!/bin/bash
# sbatch --export=ALL,MODEL=medgemma job_stageA.sh : counterfactuals (test) + hidden states (all splits) on vqa_rad + slake
#SBATCH --job-name=cvlm_stageA
#SBATCH --partition=GPU-A100
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True LC_ALL=C PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TMPDIR=$HOME/tmp_pip
PY=~/miniconda3/envs/countervlm/bin/python
echo "host $(hostname) job ${SLURM_JOB_ID} model ${MODEL} started $(date)"
for DS in ${DATASETS:-vqa_rad slake}; do
  $PY -u scripts/run_counterfactual.py --model "$MODEL" --dataset "$DS" --splits test --batch_size 16
  $PY -u scripts/extract_hidden.py --model "$MODEL" --dataset "$DS" --batch_size 8
done
echo "finished $(date)"
