#!/bin/bash
# Revised experiment (offline, reuses results/raw + features). sbatch job_revised.sh
#SBATCH --job-name=cvlm_revised
#SBATCH --partition=GPU-A100
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
set -euo pipefail
export LC_ALL=C PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TMPDIR=$HOME/tmp_pip
PY=~/miniconda3/envs/countervlm/bin/python
echo "host $(hostname) job ${SLURM_JOB_ID} started $(date)"
STEPS=${STEPS:-canonical train evaluate judge figures}
for S in $STEPS; do
  echo "=== step $S $(date)"
  case $S in
    canonical) $PY -u scripts/build_canonical_matrix.py ;;
    train)     $PY -u scripts/train_gated.py ${TRAIN_ARGS:-} ;;
    evaluate)  $PY -u scripts/evaluate_revised.py ;;
    judge)     $PY -u scripts/llm_judge.py ;;
    figures)   $PY -u scripts/make_revised_figures.py ;;
  esac
done
echo "finished $(date)"
