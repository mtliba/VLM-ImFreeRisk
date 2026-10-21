#!/bin/bash
#SBATCH --job-name=cvlm_samp_fix
#SBATCH --partition=GPU-A100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TMPDIR=$HOME/tmp_pip PYTHONUNBUFFERED=1 LC_ALL=C
PY=~/miniconda3/envs/countervlm/bin/python
for M in lingshu7b; do for DS in pmcvqa omnimedvqa; do
  $PY -u scripts/run_sampling.py --model $M --dataset $DS --splits test --k 5 --batch_size 2
done; done
echo finished
