#!/bin/bash
# Stage 2: features -> selector (3 seeds, ablations) -> evaluation -> figures. sbatch job_stage2.sh
#SBATCH --job-name=cvlm_stage2
#SBATCH --partition=GPU-A100
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
set -euo pipefail
export LC_ALL=C PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TMPDIR=$HOME/tmp_pip
PY=~/miniconda3/envs/countervlm/bin/python
echo "host $(hostname) job ${SLURM_JOB_ID} started $(date)"
STEPS=${STEPS:-oracle features selector evaluate figures}
for S in $STEPS; do
  echo "=== step $S $(date)"
  case $S in
    oracle)   $PY -u scripts/oracle_analysis.py ;;
    features) $PY -u scripts/extract_biomedclip.py --encoder biomedclip ${SPLITS:+--splits $SPLITS} ${MODELS:+--models $MODELS}
              $PY -u scripts/extract_biomedclip.py --encoder clip ${SPLITS:+--splits $SPLITS} ${MODELS:+--models $MODELS} ;;
    selector) $PY -u scripts/train_selector.py --encoder biomedclip ${MODELS:+--models $MODELS}
              $PY -u scripts/train_selector.py --encoder clip --feats iqy ${MODELS:+--models $MODELS} ;;
    evaluate) $PY -u scripts/evaluate_selector.py ;;
    figures)  $PY -u scripts/make_figures.py ;;
  esac
done
echo "finished $(date)"
