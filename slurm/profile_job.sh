#!/bin/bash
#SBATCH --job-name=profile_mc
#SBATCH --partition=cpubase_bycore_b1
#SBATCH --cpus-per-task=2
#SBATCH --mem=3G
#SBATCH --time=00:30:00
#SBATCH --output=results/logs/profile_job_%j.out
#SBATCH --error=results/logs/profile_job_%j.err

# profile_job.sh
# CPU profiling and summary (time + gprof + SLURM)
# Output: results/profile_YYYYMMDD_HHMMSS/

set -Eeuo pipefail

TIME_BIN="/cvmfs/soft.computecanada.ca/gentoo/2023/x86-64-v3/usr/bin/time"

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
LOG_DIR="${SUBMIT_DIR}/results/logs"
mkdir -p "$LOG_DIR"
BIN="${SUBMIT_DIR}/bin/monte_carlo"
RESULTS_DIR="${SUBMIT_DIR}/results/profile_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

RUN="srun --ntasks=1 --cpus-per-task=${OMP_NUM_THREADS} --cpu-bind=cores"

echo "CPU Profiling (time + gprof)"
echo "Time: $(date)"
echo "Cores: ${OMP_NUM_THREADS}"
echo "Results: $RESULTS_DIR"
echo ""

# PHASE 1: Baseline timing
echo "PHASE 1: Baseline timing & resource usage"
# Wall-clock time, memory usage, and CPU statistics.
# Output: baseline_time.txt
$TIME_BIN -v $RUN "$BIN" > "$RESULTS_DIR/baseline_time.txt" 2>&1
echo "baseline_time.txt"

# PHASE 2: gprof run (generate gmon.out)
echo "PHASE 2: gprof instrumentation run"
# Function-level profiling with gprof.
# Output: gmon.out
$RUN "$BIN" > "$RESULTS_DIR/program_output.txt" 2>&1 || true

if [[ -f "gmon.out" ]]; then
    mv gmon.out "$RESULTS_DIR/gmon.out"
    echo "gmon.out"
else
    echo "WARNING: gmon.out not found (was the binary built with -pg?)"
fi

# PHASE 3: convert gmon.out to report
echo "PHASE 3: gprof report"
if [[ -f "$RESULTS_DIR/gmon.out" ]]; then
    gprof "$BIN" "$RESULTS_DIR/gmon.out" > "$RESULTS_DIR/gprof_report.txt" 2>&1
    echo "gprof_report.txt"
else
    echo "Skipping gprof (no gmon.out)"
fi

# PHASE 4: SLURM accounting logs
echo "PHASE 4: SLURM job logs"
# Job resource usage and allocation details from SLURM.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    sacct -j "$SLURM_JOB_ID" --format=JobID,JobName,Partition,AllocCPUS,State,Elapsed,MaxRSS \
        > "$RESULTS_DIR/slurm_acct.txt" 2>&1
    echo "slurm_acct.txt"
else
    echo "SLURM_JOB_ID not set (running interactively)"
fi

echo ""
echo "Completed CPU profiling."
echo "Results in: $RESULTS_DIR"
echo "Key files:"
echo "baseline_time.txt       (wall time, memory from /usr/bin/time -v)"
echo "gprof_report.txt        (function-level profiling from gprof)"
echo "slurm_acct.txt          (resource allocation / accounting, if under SLURM)"