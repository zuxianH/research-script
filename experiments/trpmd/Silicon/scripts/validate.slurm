#!/bin/bash
# Submit from Silicon/: sbatch --account=YOUR_CPU_ACCOUNT scripts/validate.slurm
#SBATCH --job-name=si_trpmd_smoke
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=results/smoke-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the Silicon directory}"
source environment/env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export REPLICATE='2 2 2' RUNSTEPS=100 IPI_TIMESTEP_FS=0.5
export THERMO_TAU_FS=1000000 MONITOR_TIMEOUT_S=100
echo "Host: $(hostname); job: $SLURM_JOB_ID"
"$PYTHON_BIN" --version
"$PYTHON_BIN" -m pip freeze
for beads in 1 5; do
  export NUM_BEADS=$beads NUM_CLIENT=$beads
  export RUN_NAME="smoke_${SLURM_JOB_ID}_b${beads}"
  bash scripts/run.sh
done
