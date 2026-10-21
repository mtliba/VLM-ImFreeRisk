#!/bin/bash
#SBATCH --job-name=cvlm_feats
#SBATCH --partition=GPU-A100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
set -uo pipefail
export TMPDIR=$HOME/tmp_pip PYTHONUNBUFFERED=1 LC_ALL=C
PY=~/miniconda3/envs/countervlm/bin/python
$PY -u scripts/extract_biomedclip.py --encoder biomedclip --datasets vqa_rad slake --models gemma3 qwen3b qwen2vl2b lingshu7b internvl3_8b qwen3vl8b
$PY -u scripts/extract_biomedclip.py --encoder biomedclip --datasets pathvqa pmcvqa omnimedvqa --models medgemma gemma3 qwen qwen3b qwen2vl2b lingshu7b internvl3_8b qwen3vl8b
echo finished
