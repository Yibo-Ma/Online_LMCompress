#!/bin/bash
#SBATCH -J olmc_sweep
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
#SBATCH --export=ALL
#
# One array task == one parameter combination (see experiments/sweep.py gen).
# Submit with the array range `sweep.py gen` prints, e.g.:
#
#   sbatch --array=0-242%16 experiments/run_array.sbatch results/sweeps/text/<group>
#
# Resubmitting the same array only re-runs rows missing result.json (--skip-done).

set -euo pipefail

# ---- cluster-specific: edit these once -----------------------------------
PROJECT_DIR="/mnt/inaisfs/data/home/yibo_uw/Research/LLMCompression/online_LLMCompression"
CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="llmcompression"
# --------------------------------------------------------------------------

GROUP_DIR="${1:?usage: sbatch --array=0-N%K experiments/run_array.sbatch <group_dir>}"

cd "$PROJECT_DIR"
mkdir -p logs

source "$CONDA_SH"
conda activate "$CONDA_ENV"

export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8      # required for deterministic cuBLAS
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_ENDPOINT="${HF_HUB_ENDPOINT:-https://hf-mirror.com}"

python experiments/sweep.py run \
    --group "$GROUP_DIR" \
    --index "${SLURM_ARRAY_TASK_ID:-0}" \
    --skip-done
