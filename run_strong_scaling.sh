#!/bin/bash
# Strong scaling experiment - fixed M, varying threads

# Thread counts to test
THREADS=(1 2 4 8 16 32)

# Clear previous results
rm -f results/simulation_results.csv

echo "Starting strong scaling experiments..."
echo "Fixed problem size: M = 100,000,000"
echo ""

for t in "${THREADS[@]}"; do
    echo "=========================================="
    echo "Running with $t thread(s)..."
    echo "=========================================="

    export OMP_NUM_THREADS=$t

    # Run the simulation
    ./bin/monte_carlo

    echo "Completed $t thread(s)"
    echo ""
done

echo "=========================================="
echo "All experiments complete!"
echo "=========================================="

# Copy results for analysis
cp results/simulation_results.csv results/a.csv

# Run analysis
echo "Running analysis..."
python3 experiments/analysis/scaling_analysis.py

echo "Done! Check experiments/analysis/plots/strong_scaling.png"
