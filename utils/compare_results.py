#!/usr/bin/env python3
"""
Script to compare results between openmp_opt and openmp
Calculates averages and improvements (speedup/improvements)
"""

import pandas as pd
import numpy as np

def compare_implementations(csv_file):
    """
    Compare openmp_opt vs openmp implementations
    """
    # Read CSV
    df = pd.read_csv(csv_file)

    # Filter data by model
    openmp_opt = df[df['model_name'] == 'openmp_opt']
    openmp = df[df['model_name'] == 'openmp']

    print("=" * 80)
    print("COMPARISON: openmp_opt vs openmp")
    print("=" * 80)
    print(f"\nNumber of executions:")
    print(f"  openmp_opt: {len(openmp_opt)}")
    print(f"  openmp:     {len(openmp)}")

    # Metrics to compare
    metrics = {
        'execution_time_ms': 'Total execution time (ms)',
        'MC_throughput_secs': 'MC Throughput (sims/sec)',
        'kernel_time_ms': 'Kernel time (ms)',
        'overhead_time_ms': 'Overhead time (ms)',
        'kernel_throughput': 'Kernel throughput'
    }

    print("\n" + "=" * 80)
    print("AVERAGES BY IMPLEMENTATION")
    print("=" * 80)

    results = {}

    for metric, description in metrics.items():
        if metric in df.columns:
            avg_opt = openmp_opt[metric].mean()
            avg_openmp = openmp[metric].mean()
            std_opt = openmp_opt[metric].std()
            std_openmp = openmp[metric].std()

            results[metric] = {
                'openmp_opt_avg': avg_opt,
                'openmp_opt_std': std_opt,
                'openmp_avg': avg_openmp,
                'openmp_std': std_openmp
            }

            print(f"\n{description}:")
            print(f"  openmp_opt: {avg_opt:>15.2f} ± {std_opt:>10.2f}")
            print(f"  openmp:     {avg_openmp:>15.2f} ± {std_openmp:>10.2f}")

    print("\n" + "=" * 80)
    print("IMPROVEMENTS (SPEEDUP/IMPROVEMENT)")
    print("=" * 80)

    # For times: lower is better (speedup = time_openmp / time_opt)
    # For throughput: higher is better (speedup = throughput_opt / throughput_openmp)

    time_metrics = ['execution_time_ms', 'kernel_time_ms', 'overhead_time_ms']
    throughput_metrics = ['MC_throughput_secs', 'kernel_throughput']

    print("\nTimes (lower is better):")
    for metric in time_metrics:
        if metric in results:
            avg_opt = results[metric]['openmp_opt_avg']
            avg_openmp = results[metric]['openmp_avg']
            speedup = avg_openmp / avg_opt if avg_opt != 0 else float('inf')
            improvement_pct = ((avg_openmp - avg_opt) / avg_openmp) * 100 if avg_openmp != 0 else 0.0

            print(f"\n  {metrics[metric]}:")
            print(f"    Speedup: {speedup:.2f}x")
            print(f"    Improvement: {improvement_pct:.2f}% faster")

    print("\n" + "-" * 80)
    print("\nThroughput (higher is better):")
    for metric in throughput_metrics:
        if metric in results:
            avg_opt = results[metric]['openmp_opt_avg']
            avg_openmp = results[metric]['openmp_avg']
            speedup = avg_opt / avg_openmp if avg_openmp != 0 else float('inf')
            improvement_pct = ((avg_opt - avg_openmp) / avg_openmp) * 100 if avg_openmp != 0 else 0.0
            print(f"\n  {metrics[metric]}:")
            print(f"    Speedup: {speedup:.2f}x")
            print(f"    Improvement: {improvement_pct:.2f}% higher")

    # Additional statistical summary
    print("\n" + "=" * 80)
    print("DETAILED STATISTICAL SUMMARY")
    print("=" * 80)

    print("\n--- openmp_opt ---")
    print(openmp_opt[list(metrics.keys())].describe())

    print("\n--- openmp ---")
    print(openmp[list(metrics.keys())].describe())


if __name__ == "__main__":
    csv_file = "../results/simulation_results.csv"
    compare_implementations(csv_file)
