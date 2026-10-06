#!/bin/bash
#SBATCH --job-name=audiotwin_smoke
#SBATCH --array=0-1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:20:00
#SBATCH --output=logs/slurm-%x-%A_%a.out
#SBATCH --error=logs/slurm-%x-%A_%a.err

set -eo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

source /etc/profile 2>/dev/null || true
module load python/3.11.7-gcc-12.2.0

PYTHON="python"
if [ -x .venv/bin/python ]; then
  PYTHON=".venv/bin/python"
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
OUT_DIR="${AUDIOTWIN_OUT_DIR:-results/hpc_smoke}"

echo "AudioTwin smoke task ${TASK_ID} on $(hostname) using $(${PYTHON} --version 2>&1)"
"${PYTHON}" -m experiments.runner \
  task_id="${TASK_ID}" \
  out_dir="${OUT_DIR}" \
  overwrite=true \
  n_listeners=1 \
  'contexts=[noisy_conversation]' \
  'families=[balanced]' \
  'methods=[prescription,audiotwin_random]' \
  n_queries=3 \
  n_candidates=32 \
  compute_optimum=false
