#!/bin/bash
#SBATCH --job-name=cu-ni-compositions
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=cu-ni-compositions-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this script from the GDPy repository root}"
export OMP_NUM_THREADS=1

# Activate an environment with GDPy installed before submitting, or add your
# cluster's environment/module setup here. Both searches use this allocation.
gdp -d run-cu-ni-compositions \
  --runtime examples/global_optimisation/runtimes/emt.yaml \
  explore examples/global_optimisation/explorations/basin_hopping/cu_ni_compositions.yaml
