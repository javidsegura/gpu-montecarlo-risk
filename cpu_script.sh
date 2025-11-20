#!/bin/bash
#SBATCH --job-name=montecarlo_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --mem=3GB
#SBATCH --output=slurm-cpu-%j.out
#SBATCH --error=slurm-cpu-%j.err

set -e

echo "=========================================="
echo "Monte Carlo Risk Analysis (CPU)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Date: $(date)"
echo "=========================================="

# Load required modules
module purge
module load StdEnv/2023
module load gcc/12.3
module load cuda/12.2
module load python/3.11.5
module load gsl/2.7
module load libyaml/0.2.5

echo ""
echo "=== Loaded Modules ==="
module list
echo "======================"
echo ""

export OMP_NUM_THREADS=2
export OMP_CANCELLATION=true
echo "OpenMP threads: $OMP_NUM_THREADS"
echo "OpenMP cancellation: $OMP_CANCELLATION"
echo ""

# Change to the submission directory
cd "$SLURM_SUBMIT_DIR" || { echo "Failed to change to submit directory"; exit 1; }
echo "Working directory: $(pwd)"
echo ""

# STEP 1: Setup Python virtual environment
echo "=========================================="
echo "STEP 1: Setting up Python environment"
echo "=========================================="
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi
source venv/bin/activate

# Install dependencies
echo "Installing Python dependencies..."
pip install --no-cache-dir --upgrade pip
pip install --no-cache-dir -r minimal_req.txt
echo ""

# STEP 2: Preprocessing
echo "=========================================="
echo "STEP 2: Running preprocessing"
echo "=========================================="
cd preprocessing || { echo "Failed to enter preprocessing directory"; exit 1; }
python3 main.py
cd .. || { echo "Failed to return to root directory"; exit 1; }
echo ""

# STEP 3: Compile C programs
echo "=========================================="
echo "STEP 3: Compiling programs"
echo "=========================================="
make clean
make all

# Verify binary was created
if [ ! -f "bin/monte_carlo" ]; then
    echo "Error: Compilation failed - binary not found"
    exit 1
fi
echo "Compilation successful!"
echo ""

# STEP 4: Run Python Serial (optional)
echo "=========================================="
echo "STEP 4: Python Serial (Optional)"
echo "=========================================="
if [ -f "src/01-python-serial/python_serial.py" ]; then
    python3 src/01-python-serial/python_serial.py
else
    echo "Skipping Python serial (file not found)"
fi
echo ""

# STEP 5: Run C simulations
echo "=========================================="
echo "STEP 5: Running C/OpenMP simulations"
echo "=========================================="
./bin/monte_carlo
echo ""

# STEP 6: Display results
echo "=========================================="
echo "STEP 6: Results Summary"
echo "=========================================="
if [ -f "results/simulation_results.csv" ]; then
    echo "Last 5 simulation results:"
    tail -n 5 results/simulation_results.csv
else
    echo "Warning: Results file not found"
fi
echo ""

echo "=========================================="
echo "CPU Simulation Complete!"
echo "Job finished at: $(date)"
echo "=========================================="

deactivate

