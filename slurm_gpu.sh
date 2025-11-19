#!/bin/bash
#SBATCH --job-name=montecarlo_cuda
#SBATCH --partition=gpu-node
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --mem=16GB
#SBATCH --output=slurm-cuda-%j.out
#SBATCH --error=slurm-cuda-%j.err

set -e

echo "=========================================="
echo "Monte Carlo CUDA GPU Simulation (1 Node)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NODELIST"
echo "Number of nodes: $SLURM_NNODES"
echo "Date: $(date)"
echo "Partition: $SLURM_JOB_PARTITION"
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

# Display GPU information on all nodes
echo "=== GPU Information (All Nodes) ==="
srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 bash -c 'echo "Node: $(hostname)"; nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader'
echo "===================================="
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

# STEP 3: Compile C/CUDA programs
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

# STEP 4: Run CUDA simulation
echo "=========================================="
echo "STEP 4: Running CUDA simulation"
echo "=========================================="
echo "Note: CUDA implementation runs on single GPU per execution"
echo "For multi-GPU, run multiple instances or use MPI-GPU version"
./bin/monte_carlo
echo ""

# STEP 5: Display results
echo "=========================================="
echo "STEP 5: Results Summary"
echo "=========================================="
if [ -f "results/simulation_results.csv" ]; then
    echo "Last 5 simulation results:"
    tail -n 5 results/simulation_results.csv
else
    echo "Warning: Results file not found"
fi
echo ""

echo "=========================================="
echo "CUDA Simulation Complete!"
echo "Job finished at: $(date)"
echo "=========================================="

deactivate

