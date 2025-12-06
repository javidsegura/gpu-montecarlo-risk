#!/bin/bash
# Weak scaling experiment - M scales with threads

# Base problem size per thread
BASE_M=10000000

# Thread counts to test
THREADS=(1 2 4 8 16 32)

# Clear previous results
rm -f results/simulation_results.csv

echo "Starting weak scaling experiments..."
echo "Base problem size per thread: $BASE_M"
echo ""

for t in "${THREADS[@]}"; do
    M=$((BASE_M * t))

    echo "=========================================="
    echo "Running with $t thread(s), M = $M"
    echo "=========================================="

    export OMP_NUM_THREADS=$t

    # Update M in config.yaml
    sed -i.bak "s/^M: .*/M: $M/" config.yaml

    # Run the simulation
    ./bin/monte_carlo

    echo "Completed $t thread(s)"
    echo ""
done

# Restore original config
if [ -f config.yaml.bak ]; then
    mv config.yaml.bak config.yaml
fi

echo "=========================================="
echo "All experiments complete!"
echo "=========================================="

# Run analysis
echo "Running analysis..."
python3 experiments/analysis/weak_scaling_analysis.py

echo "Done! Check experiments/analysis/plots/weak_scaling.png"
