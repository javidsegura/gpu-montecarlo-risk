#!/bin/bash
#SBATCH --job-name=montecarlo_risk
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --mem=8GB
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

echo "Monte Carlo Risk Analysis Pipeline"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Date: $(date)"

# Load required modules
module purge

module load StdEnv/2023
module load gcc/12.3
module load python/3.11.5
module load gsl/2.7
module load libyaml/0.2.5

module list

export OMP_NUM_THREADS=4
echo "OpenMP threads: $OMP_NUM_THREADS"

cd $SLURM_SUBMIT_DIR
echo "Working directory: $(pwd)"


echo "STEP 1: Preprocessing"
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

# Install dependencies
pip install --no-cache-dir --upgrade pip
pip install --no-cache-dir -r minimal_req.txt

# Run preprocessing
cd preprocessing
python3 main.py
cd ..



echo "STEP 2: Compiling C Programs"
make clean
make all



echo "STEP 3: Python Serial"
python3 src/01-python-serial/python_serial.py


echo "STEP 4: C Serial"
./bin/monte_carlo


echo "COMPLETED!"
deactivate