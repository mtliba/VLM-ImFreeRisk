#!/bin/bash
# sbatch --export=ALL,MODEL=medgemma,SPLITS=validation job_verify.sh
#SBATCH --job-name=cvlm_verify
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
echo "host $(hostname) job ${SLURM_JOB_ID} model ${MODEL} splits ${SPLITS} started $(date)"
for DS in ${DATASETS:-vqa_rad slake}; do
  $PY -u scripts/run_verify.py --model "$MODEL" --dataset "$DS" --splits $SPLITS --batch_size "${BS:-16}" ${LIMIT:+--limit $LIMIT}
done
echo "finished $(date)"
