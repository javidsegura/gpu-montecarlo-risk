#!/usr/bin/env python3
"""
summarize_profile.py

Generate a high-level profiling summary from profile_job.sh outputs.

- Reads:
    baseline_time.txt   (PHASE 1, /bin/time -v)
    perf_stat.txt       (PHASE 3, perf stat -d -r 3)
    slurm_acct.txt      (PHASE 4, sacct summary)  [optional]

- Writes:
    summary.txt        
"""

import sys
import re
from pathlib import Path


# perf stat parsing 

def parse_perf_stat(filepath: str) -> dict:
    """Extract metrics from perf stat output."""
    metrics = {}
    try:
        with open(filepath, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"WARNING: {filepath} not found")
        return metrics

    patterns = {
        'cycles': r'(\d+(?:,\d+)*)\s+cycles',
        'instructions': r'(\d+(?:,\d+)*)\s+instructions',
        'cache_misses': r'(\d+(?:,\d+)*)\s+cache-misses',
        'cache_references': r'(\d+(?:,\d+)*)\s+cache-references',
        'context_switches': r'(\d+(?:,\d+)*)\s+context-switches',
        'page_faults': r'(\d+(?:,\d+)*)\s+page-faults',
        'dTLB_loads': r'(\d+(?:,\d+)*)\s+dTLB-loads',
        'dTLB_misses': r'(\d+(?:,\d+)*)\s+dTLB-load-misses',
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, content)
        if match:
            num_str = match.group(1).replace(',', '')
            try:
                metrics[key] = int(num_str)
            except ValueError:
                pass

    return metrics


def compute_metrics(metrics: dict) -> dict:
    """Compute derived metrics (IPC, miss rates, etc.) from raw counters."""
    cycles = metrics.get('cycles', 1)
    instructions = metrics.get('instructions', 1)
    cache_misses = metrics.get('cache_misses', 0)
    cache_refs = metrics.get('cache_references', 1)
    cs = metrics.get('context_switches', 0)
    pf = metrics.get('page_faults', 0)
    dtlb_misses = metrics.get('dTLB_misses', 0)
    dtlb_loads = metrics.get('dTLB_loads', 1)

    ipc = instructions / cycles if cycles > 0 else 0.0
    cache_miss_rate = (cache_misses / cache_refs * 100.0) if cache_refs > 0 else 0.0
    dtlb_miss_rate = (dtlb_misses / dtlb_loads * 100.0) if dtlb_loads > 0 else 0.0
    cs_rate = (cs / cycles) * 1e6 if cycles > 0 else 0.0

    return {
        'ipc': ipc,
        'cache_miss': cache_miss_rate,
        'dtlb_miss': dtlb_miss_rate,
        'context_switches': cs_rate,
        'page_faults': pf,
    }


#  baseline_time.txt parsing ( /bin/time -v ) 
def parse_baseline_time(filepath: str) -> dict:
    """
    Parse /bin/time -v output for key fields:
      - Elapsed (wall clock) time
      - User time (seconds)
      - System time (seconds)
      - Maximum resident set size (kbytes)
    """
    info = {
        "elapsed": None,
        "user_time": None,
        "sys_time": None,
        "max_rss_kb": None,
    }

    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith("Elapsed (wall clock) time"):
                    # e.g. "Elapsed (wall clock) time (h:mm:ss or m:ss):  0:12.34"
                    value = line.split(":", 1)[-1].strip()
                    info["elapsed"] = value
                elif line.startswith("User time (seconds)"):
                    value = line.split(":", 1)[-1].strip()
                    try:
                        info["user_time"] = float(value)
                    except ValueError:
                        pass
                elif line.startswith("System time (seconds)"):
                    value = line.split(":", 1)[-1].strip()
                    try:
                        info["sys_time"] = float(value)
                    except ValueError:
                        pass
                elif line.startswith("Maximum resident set size (kbytes)"):
                    value = line.split(":", 1)[-1].strip()
                    try:
                        info["max_rss_kb"] = int(value)
                    except ValueError:
                        pass
    except FileNotFoundError:
        print(f"WARNING: {filepath} not found")

    return info

# summary generation

def generate_summary(results_dir: Path,
                     baseline_info: dict,
                     perf_metrics_raw: dict,
                     perf_metrics_derived: dict,
                     output_file: Path) -> None:
    """Write overall profiling summary to summary.txt."""

    lines = []
    lines.append("PROFILE SUMMARY")
    lines.append(f"Results directory: {results_dir}")
    lines.append("")

    # Timing & memory (baseline_time.txt)
    lines.append("TIMING & MEMORY (from baseline_time.txt)")
    if any(v is not None for v in baseline_info.values()):
        if baseline_info.get("elapsed") is not None:
            lines.append(f"  Elapsed (wall clock):       {baseline_info['elapsed']}")
        if baseline_info.get("user_time") is not None:
            lines.append(f"  User time (s):              {baseline_info['user_time']:.3f}")
        if baseline_info.get("sys_time") is not None:
            lines.append(f"  System time (s):            {baseline_info['sys_time']:.3f}")
        if baseline_info.get("max_rss_kb") is not None:
            lines.append(f"  Max RSS (kB):               {baseline_info['max_rss_kb']:,}")
    else:
        lines.append("  (baseline_time.txt not found or not parsed)")
    lines.append("")

    # Perf counters (perf_stat.txt)
    lines.append("CPU PERFORMANCE COUNTERS (from perf_stat.txt)")
    if perf_metrics_raw:
        lines.append(f"  Cycles:                     {perf_metrics_raw.get('cycles', 0):>15,}")
        lines.append(f"  Instructions:               {perf_metrics_raw.get('instructions', 0):>15,}")
        lines.append(f"  Cache Misses:               {perf_metrics_raw.get('cache_misses', 0):>15,}")
        lines.append(f"  Cache References:           {perf_metrics_raw.get('cache_references', 0):>15,}")
        lines.append(f"  dTLB Load Misses:           {perf_metrics_raw.get('dTLB_misses', 0):>15,}")
        lines.append(f"  dTLB Loads:                 {perf_metrics_raw.get('dTLB_loads', 0):>15,}")
        lines.append(f"  Context Switches:           {perf_metrics_raw.get('context_switches', 0):>15,}")
        lines.append(f"  Page Faults:                {perf_metrics_raw.get('page_faults', 0):>15,}")
        lines.append("")
        lines.append("  Derived metrics:")
        lines.append(f"    IPC (instructions/cycle):      {perf_metrics_derived['ipc']:10.2f}")
        lines.append(f"    Cache miss rate:               {perf_metrics_derived['cache_miss']:10.1f} %")
        lines.append(f"    dTLB miss rate:                {perf_metrics_derived['dtlb_miss']:10.1f} %")
        lines.append(f"    Context switches / 1M cycles:  {perf_metrics_derived['context_switches']:10.0f}")
        lines.append(f"    Page faults (count):           {perf_metrics_derived['page_faults']:10.0f}")
    else:
        lines.append("  (perf_stat.txt not found or no metrics parsed)")
    lines.append("")

    # Interpretation hints 
    lines.append("INTERPRETATION NOTES")
    lines.append("  - Higher IPC generally indicates better core utilization.")
    lines.append("  - High cache/TLB miss rates usually indicate memory hierarchy pressure.")
    lines.append("  - Many context switches/page faults often correlate with OS or I/O overhead.")
    lines.append("  - For rigorous analysis, compare these metrics across runs,")
    lines.append("    problem sizes, compilers, and node types.")
    lines.append("")

    # References
    lines.append("REFERENCES TO RAW DATA FILES")
    lines.append("  - baseline_time.txt   : full /bin/time -v output")
    lines.append("  - perf_report.txt     : function-level hotspots and call stacks (perf report)")
    lines.append("  - perf_stat.txt       : full perf stat output with hardware counters")
    lines.append("  - slurm_acct.txt      : SLURM accounting (allocation, MaxRSS, elapsed)")
    lines.append("")

    report_text = "\n".join(lines)

    with open(output_file, 'w') as f:
        f.write(report_text)

    print(f"Summary written to: {output_file}")
    print(report_text)


# main 

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 summarize_profile.py <results_dir>")
        sys.exit(1)

    results_dir = Path(sys.argv[1])

    baseline_file = results_dir / "baseline_time.txt"
    perf_stat_file = results_dir / "perf_stat.txt"
    slurm_acct_file = results_dir / "slurm_acct.txt"  
    summary_file = results_dir / "summary.txt"

    # Parse baseline timing
    baseline_info = parse_baseline_time(str(baseline_file))

    # Parse perf stat metrics
    perf_metrics_raw = parse_perf_stat(str(perf_stat_file))
    perf_metrics_derived = compute_metrics(perf_metrics_raw) if perf_metrics_raw else {
        'ipc': 0.0,
        'cache_miss': 0.0,
        'dtlb_miss': 0.0,
        'context_switches': 0.0,
        'page_faults': 0,
    }

    # Generate summary
    generate_summary(results_dir, baseline_info, perf_metrics_raw, perf_metrics_derived, summary_file)


if __name__ == "__main__":
    main()
