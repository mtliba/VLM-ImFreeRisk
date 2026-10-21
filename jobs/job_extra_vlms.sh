#!/bin/bash
# sbatch --export=ALL,MODEL=qwen3b job_extra_vlms.sh : KEEP on all splits + 3 interventions on test; sampling for MODEL on val+test
#SBATCH --job-name=cvlm_extra
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
echo "host $(hostname) job ${SLURM_JOB_ID} model ${MODEL} mode ${MODE:-full} started $(date)"
if [ "${MODE:-full}" = "sampling" ]; then
  for DS in vqa_rad slake; do $PY -u scripts/run_sampling.py --model "$MODEL" --dataset "$DS" --splits validation test --k 5 --batch_size 8; done
else
  for DS in vqa_rad slake; do
    $PY -u scripts/run_vlm.py --model "$MODEL" --dataset "$DS" --splits test validation train --actions keep --batch_size 16
    $PY -u scripts/run_vlm.py --model "$MODEL" --dataset "$DS" --splits test --actions reprompt enhance zoom --batch_size 16
  done
  for DS in vqa_rad slake; do $PY -u scripts/run_sampling.py --model "$MODEL" --dataset "$DS" --splits test --k 5 --batch_size 8; done
fi
echo "finished $(date)"
