#!/bin/bash
#SBATCH --job-name=audiotwin_pilot
#SBATCH --array=0-31
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
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
OUT_DIR="${AUDIOTWIN_OUT_DIR:-results/audiotwin_pilot}"

echo "AudioTwin pilot task ${TASK_ID} on $(hostname) using $(${PYTHON} --version 2>&1)"
"${PYTHON}" -m experiments.runner \
  task_id="${TASK_ID}" \
  out_dir="${OUT_DIR}" \
  n_listeners=1 \
  'contexts=[noisy_conversation,lecture,music]' \
  'families=[intelligibility,comfort,balanced,noise_sensitive]' \
  'methods=[prescription,prescription_linear,random_feasible,audiotwin_random]' \
  n_queries=12 \
  n_candidates=512 \
  compute_optimum=true \
  optimum.n_random=400 \
  optimum.maxiter=60 \
  optimum.popsize=12
