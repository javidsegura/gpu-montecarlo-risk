#!/bin/bash
# CPU Profiling and summary
# Output: results/profile_YYYYMMDD_HHMMSS/ with perf data + summary

#!/bin/bash
#SBATCH --job-name=profile_mc
#SBATCH --partition=cpubase_bycore_b1
#SBATCH --cpus-per-task=2
#SBATCH --mem=3G
#SBATCH --time=00:30:00
#SBATCH --output=profile_%j.out
#SBATCH --error=profile_%j.err

set -Eeuo pipefail  # Exit on error, unset variable, or failed pipe

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"         # Set working directory for the job - default to current dir
BIN="${SUBMIT_DIR}/bin/monte_carlo"            # Path to the binary to profile
RESULTS_DIR="${SUBMIT_DIR}/results/profile_$(date +%Y%m%d_%H%M%S)"  # Timestamped results directory
mkdir -p "$RESULTS_DIR"                        # Create results directory

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}   # Set OpenMP threads to match SLURM allocation
export OMP_PROC_BIND=spread                        # Spread threads across cores
export OMP_PLACES=cores                            # Bind threads to physical cores

RUN="srun --ntasks=1 --cpus-per-task=${OMP_NUM_THREADS} --cpu-bind=cores"  # SLURM run command for 1 task

echo "CPU Profiling"
echo "Time: $(date)"
echo "Cores: ${OMP_NUM_THREADS}"
echo "Results: $RESULTS_DIR"
echo ""

# PHASE 1: Baseline timing
echo "PHASE 1: Baseline timing & resource usage"
/bin/time -v $RUN "$BIN" > "$RESULTS_DIR/baseline_time.txt" 2>&1   # Measure wall time, memory, and resource usage
echo "baseline_time.txt"

# PHASE 2: CPU profiling (cycles, cache, threading, TLB)
echo "PHASE 2: CPU profiling (IPC, cache, threading, memory hierarchy)"
perf record -o "$RESULTS_DIR/perf.data" -F 4000 --call-graph dwarf -e \
    cycles,instructions,cache-misses,cache-references,\
    branch-misses,context-switches,\
    dTLB-load-misses,dTLB-loads,page-faults \
    -- $RUN "$BIN" 2>&1 | tee "$RESULTS_DIR/perf_record.log"   # Collect detailed CPU and memory events 

perf report --stdio -i "$RESULTS_DIR/perf.data" > "$RESULTS_DIR/perf_report.txt" 2>&1  # Generate human-readable perf report
echo "perf.data, perf_report.txt"

# PHASE 3: Aggregated statistics (3 runs for reliability)
echo "PHASE 3: Aggregated statistics (3 runs averaged)"
perf stat -d -r 3 -- $RUN "$BIN" > "$RESULTS_DIR/perf_stat.txt" 2>&1   # Collect hardware counters, averaged over 3 runs
echo "perf_stat.txt"


# PHASE 4: SLURM accounting logs
echo "PHASE 4: SLURM job logs"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    sacct -j "$SLURM_JOB_ID" --format=JobID,JobName,Partition,AllocCPUS,State,Elapsed,MaxRSS > "$RESULTS_DIR/slurm_acct.txt" 2>&1  # Extract SLURM job logs
    echo "slurm_acct.txt"
else
    echo "SLURM_JOB_ID not set (running interactively)"
fi


# PHASE 5: Summary report
echo "PHASE 5: Summary report"
python3 slurm/summarize_profile.py "$RESULTS_DIR" 2>&1   # Generate high-level summary
echo "summary.txt"

echo ""
echo "Completed CPU profiling."
echo "Results in: $RESULTS_DIR"
echo "Key files:"
echo "baseline_time.txt         (wall time, memory from /bin/time -v)"
echo "perf_report.txt           (hotspot functions)"
echo "perf_stat.txt             (raw perf counters, 3-run average)"
echo "slurm_acct.txt            (resource allocation / accounting)"
echo "summary.txt               (high-level summary across the above)"