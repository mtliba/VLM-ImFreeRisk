#!/bin/bash
# One VLM on the new benchmarks (pathvqa pmcvqa omnimedvqa): same protocol as job_stageB1.sh
#SBATCH --job-name=cvlm_B2
#SBATCH --partition=GPU-A100
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True LC_ALL=C PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TMPDIR=$HOME/tmp_pip
PY=~/miniconda3/envs/countervlm/bin/python
echo "host $(hostname) job ${SLURM_JOB_ID} model ${MODEL} started $(date)"
for DS in ${DATASETS:-pathvqa pmcvqa omnimedvqa}; do
  while [ ! -f data/$DS/test.jsonl ]; do echo "waiting for data/$DS/test.jsonl"; sleep 300; done
  $PY -u scripts/run_vlm.py --model "$MODEL" --dataset "$DS" --splits test validation train --actions keep --batch_size "${BS:-16}"
  $PY -u scripts/run_vlm.py --model "$MODEL" --dataset "$DS" --splits test --actions reprompt enhance zoom --batch_size "${BS:-16}"
  $PY -u scripts/run_sampling.py --model "$MODEL" --dataset "$DS" --splits test --k 5 --batch_size 8
  $PY -u scripts/run_counterfactual.py --model "$MODEL" --dataset "$DS" --splits test --batch_size "${BS:-16}"
  $PY -u scripts/extract_hidden.py --model "$MODEL" --dataset "$DS" --batch_size 8
done
echo "finished $(date)"
