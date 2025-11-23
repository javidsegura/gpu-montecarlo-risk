#!/bin/bash
#SBATCH --job-name=timing_openmp
#SBATCH --partition=cpubase_bycore_b1
#SBATCH --cpus-per-task=2
#SBATCH --mem=3G
#SBATCH --time=00:20:00
#SBATCH --output=timing_openmp_%j.out
#SBATCH --error=timing_openmp_%j.err

# timing_openmp.sh
# Repeated timing runs for the OpenMP implementation (no gprof).
# Assumes config.yaml is set with models: [openmp].

set -Eeuo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
BIN="${SUBMIT_DIR}/bin/monte_carlo"
RESULTS_DIR="${SUBMIT_DIR}/results/timing_openmp_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RESULTS_DIR"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

RUN="srun --ntasks=1 --cpus-per-task=${OMP_NUM_THREADS} --cpu-bind=cores"

echo "Timing-only runs for OpenMP"
echo "Time: $(date)"
echo "Cores: ${OMP_NUM_THREADS}"
echo "Results: $RESULTS_DIR"
echo ""

TIMES_FILE="$RESULTS_DIR/time_runs.txt"

for i in 1 2 3; do
    echo "-- Run $i --" | tee -a "$TIMES_FILE"
    /usr/bin/time -v $RUN "$BIN" >> "$TIMES_FILE" 2>&1
    echo "" >> "$TIMES_FILE"
done

echo ""
echo "Completed timing runs."
echo "Results in: $RESULTS_DIR"
echo "Key files:"
echo "time_runs.txt            (3x /usr/bin/time -v outputs)"
echo "results/simulation_results.csv (internal per-run timings & throughput)"
