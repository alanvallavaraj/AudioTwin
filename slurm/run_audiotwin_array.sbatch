#!/usr/bin/env bash
#SBATCH --job-name=audiotwin
#SBATCH --array=0-31
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

export PATH="/opt/slurm/bin:${PATH}"
source /etc/profile 2>/dev/null || true
module load python/3.11.7-gcc-12.2.0

if [ -d .venv ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
OUT_DIR="${AUDIOTWIN_OUT_DIR:-results/audiotwin}"

python -m experiments.runner \
  task_id="${TASK_ID}" \
  out_dir="${OUT_DIR}"
