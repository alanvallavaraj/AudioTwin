#!/bin/bash
#SBATCH --job-name=audiotwin_final_v1
#SBATCH --array=0-127%64
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-%x-%A_%a.out
#SBATCH --error=logs/slurm-%x-%A_%a.err

set -eo pipefail
source /etc/profile 2>/dev/null || true
module purge
module load python/3.11.7-gcc-12.2.0

cd /data/home/hdw517/audiotwin
mkdir -p logs results/final_v1
PYTHON=.venv/bin/python
if [[ ! -x "$PYTHON" ]]; then PYTHON=python; fi
TASK_ID=${SLURM_ARRAY_TASK_ID}

echo "AudioTwin FINAL v1 task ${TASK_ID} on $(hostname) using $(${PYTHON} --version)"

${PYTHON} -m experiments.runner \
  seed=777000111 \
  +allow_final=true \
  task_id="${TASK_ID}" \
  out_dir=results/final_v1 \
  n_listeners=1 \
  'contexts=[noisy_conversation,lecture,music]' \
  'families=[intelligibility,comfort,balanced,noise_sensitive]' \
  'methods=[prescription,prescription_linear,prescription_slow,audiotwin_qeubo,audiotwin_bald,audiotwin_gprandom]' \
  '+budgets=[0,5,10,20,30,50]' \
  compute_optimum=true \
  optimum.n_random=400 \
  optimum.maxiter=60 \
  optimum.popsize=12
