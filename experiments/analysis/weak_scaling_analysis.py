#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    data_file = Path('results/simulation_results.csv')
    output_dir = Path('experiments/analysis/plots')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading weak scaling data...")

    if not data_file.exists():
        print(f"No file found at {data_file}!")
        return

    print(f"Found data file: {data_file}")

    df = pd.read_csv(data_file)
    df = df[df['threads'] > 0]

    grouped = df.groupby('threads').agg({
        'execution_time_ms': 'mean',
        'M': 'first'
    }).reset_index().sort_values('threads')

    grouped['tts'] = grouped['execution_time_ms'] / 1000
    grouped['work_per_thread'] = grouped['M'] / grouped['threads']
    baseline_time = grouped[grouped['threads'] == 1]['tts'].values[0]
    grouped['efficiency'] = (baseline_time / grouped['tts']) * 100

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Weak Scaling Analysis', fontsize=16, fontweight='bold')

    # Plot 1: Time vs Threads (should stay flat for perfect weak scaling)
    ax = axes[0, 0]
    ax.plot(grouped['threads'], grouped['tts'], 'bo-', linewidth=2, markersize=8, label='Actual')
    ax.axhline(y=baseline_time, color='k', linestyle='--', linewidth=1, label='Ideal (constant)')
    ax.set_xlabel('Threads')
    ax.set_ylabel('Time (s)')
    ax.set_title('Time to Solution (should be flat)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Problem Size vs Threads
    ax = axes[0, 1]
    ax.plot(grouped['threads'], grouped['M'] / 1e6, 'ro-', linewidth=2, markersize=8)
    ax.set_xlabel('Threads')
    ax.set_ylabel('Problem Size (M simulations)')
    ax.set_title('Problem Size Scaling')
    ax.grid(True, alpha=0.3)

    # Plot 3: Efficiency
    ax = axes[1, 0]
    ax.plot(grouped['threads'], grouped['efficiency'], 'go-', linewidth=2, markersize=8)
    ax.axhline(y=100, color='k', linestyle='--', linewidth=1, label='Ideal')
    ax.set_xlabel('Threads')
    ax.set_ylabel('Efficiency (%)')
    ax.set_title('Weak Scaling Efficiency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 110])

    # Plot 4: Work per thread (should be constant)
    ax = axes[1, 1]
    ax.plot(grouped['threads'], grouped['work_per_thread'] / 1e6, 'mo-', linewidth=2, markersize=8)
    baseline_work = grouped[grouped['threads'] == 1]['work_per_thread'].values[0]
    ax.axhline(y=baseline_work / 1e6, color='k', linestyle='--', linewidth=1, label='Constant')
    ax.set_xlabel('Threads')
    ax.set_ylabel('Work per Thread (M sim/thread)')
    ax.set_title('Work Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'weak_scaling.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved plot to {output_dir / 'weak_scaling.png'}")

    print("\n=== Weak Scaling Summary ===")
    print(grouped[['threads', 'M', 'tts', 'efficiency']])

if __name__ == '__main__':
    main()
