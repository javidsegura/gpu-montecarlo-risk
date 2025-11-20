# SLURM Profiling Scripts
This directory contains the scripts used to profile the Monte Carlo simulation on the HPC cluster.
Profiling consists of a five-phase pipeline combining /usr/bin/time, perf, and SLURM accounting data.

## Files
### `profile_job.sh`
Main profiling script.
Runs the target binary under a controlled SLURM environment and collects all performance artifacts.

**Output:** Creates timestamped directory in `results/profile_YYYYMMDD_HHMMSS/`

### `profile_job.sh`

Generates a high-level summary (summary.txt) from selected profiling outputs
(baseline_time.txt, perf_stat.txt, slurm_acct.txt).


## Profiling Phases 

### PHASE 1:
- Collects high-level runtime and memory metrics using /usr/bin/time -v:
- Wall-clock time (total runtime)
- Maximum resident memory (MaxRSS)
- User/system CPU time
- Page faults and basic I/O statistics
**Output**: baseline_time.txt

### PHASE 2 Collects:
Collects low-level hardware events and generates a call graph showing where CPU time is spent:
- cycles
- instructions
- cache-misses / cache-references
- branch-misses
- context-switches
- dTLB-load-misses / dTLB-loads
- page-faults
**Output**: perf.data, perf_record.log, perf_report.txt

### PHASE 3: Aggregated statistics (3 runs averaged)
Runs perf stat three times to improve reliability and compute aggregate statistics:
- cycles, instructions, cache events, branches
- IPC (instructions per cycle)
- Cache miss rate
**Output**: perf_stat.txt

### PHASE 4: SLURM job logs
Extracts job-level resource usage from the SLURM scheduler:
- Job ID, name, and partition
- Allocated CPUs
- Job state
- Elapsed wall-clock time
- MaxRSS (peak memory usage)
**Output**: slurm_acct.txt

### PHASE 5: Summary
Reads outputs from previous phases and produces a consolidated overview containing:
- Key performance metrics
- Derived counters (IPC, miss rates, context switches per cycle)
- Interpretation notes
- References to raw data files
**Output**: summary.txt



## Full Output Layout
```
results/profile_YYYYMMDD_HHMMSS/
│
├── baseline_time.txt      # from Phase 1 (/bin/time -v)
├── perf.data              # from Phase 2 (perf record)
├── perf_record.log        # from Phase 2 (tee)
├── perf_report.txt        # from Phase 2 (perf report)
├── perf_stat.txt          # from Phase 3 (perf stat -d -r 3)
├── slurm_acct.txt         # from Phase 4 (sacct)
└── summary.txt            # from Phase 5 (summarize_profile.py)
```



