#!/bin/bash
#SBATCH --job-name=profile_mc
#SBATCH --partition=cpubase_bycore_b1
#SBATCH --cpus-per-task=2
#SBATCH --mem=3G
#SBATCH --time=00:30:00
#SBATCH --output=results/logs/profile_perf_%j.out
#SBATCH --error=results/logs/profile_perf_%j.err

# profile_job.sh (perf version)
# CPU profiling and summary (time + perf + SLURM)
# Output: results/profile_perf_YYYYMMDD_HHMMSS/

set -Eeuo pipefail

TIME_BIN="/cvmfs/soft.computecanada.ca/gentoo/2023/x86-64-v3/usr/bin/time"

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
LOG_DIR="${SUBMIT_DIR}/results/logs"
mkdir -p "$LOG_DIR"
BIN="${SUBMIT_DIR}/bin/monte_carlo"
RESULTS_DIR="${SUBMIT_DIR}/results/profile_perf_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

# srun que lanza el binario en el nodo de cómputo
RUN="srun --ntasks=1 --cpus-per-task=${OMP_NUM_THREADS} --cpu-bind=cores"

echo "CPU profiling with perf"
echo "Time: $(date)"
echo "Cores: ${OMP_NUM_THREADS}"
echo "Results: $RESULTS_DIR"
echo ""

# PHASE 1: baseline con time -v
echo "PHASE 1: Baseline timing"
"$TIME_BIN" -v $RUN "$BIN" > "$RESULTS_DIR/baseline_time.txt" 2>&1
echo "baseline_time.txt"

# PHASE 2: perf record (hotspots con call stacks, en el nodo de cómputo)
echo "PHASE 2: perf record (cpu-clock, call graph)"
$RUN perf record -o "$RESULTS_DIR/perf.data" \
  -F 4000 --call-graph dwarf -e cpu-clock \
  -- "$BIN" 2>&1 | tee "$RESULTS_DIR/perf_record.log"

perf report --stdio -i "$RESULTS_DIR/perf.data" > "$RESULTS_DIR/perf_report.txt" 2>&1
echo "perf_report.txt"

# PHASE 3: perf stat (solo eventos de software, también dentro de srun)
echo "PHASE 3: perf stat (software events)"
$RUN perf stat -r 3 -e task-clock,context-switches,cpu-migrations,page-faults \
  -- "$BIN" > "$RESULTS_DIR/perf_stat.txt" 2>&1
echo "perf_stat.txt"

# PHASE 4: SLURM accounting
echo "PHASE 4: SLURM job logs"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  sacct -j "$SLURM_JOB_ID" \
    --format=JobID,JobName,Partition,AllocCPUS,State,Elapsed,MaxRSS \
    > "$RESULTS_DIR/slurm_acct.txt" 2>&1
  echo "slurm_acct.txt"
else
  echo "SLURM_JOB_ID not set (running interactively)"
fi

echo ""
echo "Completed CPU profiling with perf."
echo "Results in: $RESULTS_DIR"