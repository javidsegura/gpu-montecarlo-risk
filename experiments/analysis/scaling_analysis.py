#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import glob

def main():
    data_file = Path('results/a.csv')
    output_dir = Path('experiments/analysis/plots')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading scaling data...")

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
    grouped['throughput'] = grouped['M'] / grouped['tts']
    baseline_time = grouped[grouped['threads'] == 1]['tts'].values[0]
    grouped['speedup'] = baseline_time / grouped['tts']
    grouped['efficiency'] = (grouped['speedup'] / grouped['threads']) * 100
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Strong Scaling Analysis', fontsize=16, fontweight='bold')
    
    ax = axes[0, 0]
    ax.plot(grouped['threads'], grouped['throughput'] / 1e6, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('Threads')
    ax.set_ylabel('Throughput (M sim/s)')
    ax.set_title('Throughput vs Threads')
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.plot(grouped['threads'], grouped['tts'], 'ro-', linewidth=2, markersize=8)
    ax.set_xlabel('Threads')
    ax.set_ylabel('Time (s)')
    ax.set_title('Time to Solution')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    ax.plot(grouped['threads'], grouped['speedup'], 'go-', linewidth=2, markersize=8, label='Actual')
    ax.plot(grouped['threads'], grouped['threads'], 'k--', linewidth=1, label='Ideal')
    ax.set_xlabel('Threads')
    ax.set_ylabel('Speedup')
    ax.set_title('Speedup')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    ax.plot(grouped['threads'], grouped['efficiency'], 'mo-', linewidth=2, markersize=8)
    ax.axhline(y=100, color='k', linestyle='--', linewidth=1)
    ax.set_xlabel('Threads')
    ax.set_ylabel('Efficiency (%)')
    ax.set_title('Parallel Efficiency')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 110])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'strong_scaling.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved plot to {output_dir / 'strong_scaling.png'}")
    
    print("\n=== Summary ===")
    print(grouped[['threads', 'tts', 'speedup', 'efficiency']])

if __name__ == '__main__':
    main()