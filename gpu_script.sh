#!/bin/bash
#SBATCH --job-name=montecarlo_gpu
#SBATCH --partition=gpu-node
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --time=00:10:00
#SBATCH --output=slurm-gpu-%j.out
#SBATCH --error=slurm-gpu-%j.err

set -e

echo "=========================================="
echo "Monte Carlo Risk Analysis (GPU)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Date: $(date)"
echo "=========================================="

# Load host modules for Preprocessing and Path Discovery
module purge
module load StdEnv/2023
module load python/3.11.5
module load gsl/2.7
module load libyaml/0.2.5

echo ""
echo "=== Loaded Modules (Host) ==="
module list
echo "============================="
echo ""

# Define paths
WRAP="/project/dlstack/tools/appt-gpu"
IMG="/project/dlstack/containers/tensorflow-2.16.1-gpu.sif"

# Host CUDA Path (for headers/libs missing in container)
# Obtained from 'module show cuda/12.2' on host
CUDA_HOST="/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Core/cudacore/12.2.2"

# Check for GSL and LibYAML paths
if [ -z "$EBROOTGSL" ]; then
    echo "Error: EBROOTGSL not set. GSL module might not be loaded correctly."
    exit 1
fi
if [ -z "$EBROOTLIBYAML" ]; then
    echo "Error: EBROOTLIBYAML not set. LibYAML module might not be loaded correctly."
    exit 1
fi

echo "GSL Path: $EBROOTGSL"
echo "LibYAML Path: $EBROOTLIBYAML"
echo "Host CUDA Path: $CUDA_HOST"

# Setup Container Bindings
# We bind the project directory (pwd), GSL, LibYAML, and Host CUDA
export APPTAINER_BIND="$(pwd),${EBROOTGSL},${EBROOTLIBYAML},${CUDA_HOST}:/host-cuda"

# Setup Environment Variables for Compilation inside Container
# GCC and NVCC will look at these for headers and libraries
export CPATH="${EBROOTGSL}/include:${EBROOTLIBYAML}/include:$CPATH"
export LIBRARY_PATH="${EBROOTGSL}/lib:${EBROOTLIBYAML}/lib:$LIBRARY_PATH"
export LD_LIBRARY_PATH="${EBROOTGSL}/lib:${EBROOTLIBYAML}/lib:$LD_LIBRARY_PATH"

# Change to submission directory
cd "$SLURM_SUBMIT_DIR" || exit 1
echo "Working directory: $(pwd)"

# STEP 1: Preprocessing (Host)
echo "=========================================="
echo "STEP 1: Preprocessing (Host)"
echo "=========================================="
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi
source venv/bin/activate
pip install --no-cache-dir --upgrade pip
pip install --no-cache-dir -r minimal_req.txt

echo "Running preprocessing..."
cd preprocessing
python3 main.py
cd ..
deactivate
echo "Preprocessing done."
echo ""

# STEP 2: Compilation (Container)
echo "=========================================="
echo "STEP 2: Compilation (Container)"
echo "=========================================="
# We run compilation manually because 'make' might not be in the container
# The environment variables CPATH/LIBRARY_PATH exported above will be used by gcc/nvcc inside
$WRAP "$IMG" bash -c "
set -e
echo 'Compiling inside container...'

# Create directories
mkdir -p bin obj

# Clean
rm -rf obj/* bin/*

# Compile Utilities
echo 'Compiling utilities...'
gcc -Wall -Wextra -O3 -std=gnu11 -Isrc -c src/utilities/load_binary.c -o obj/load_binary.o
gcc -Wall -Wextra -O3 -std=gnu11 -Isrc -c src/utilities/load_config.c -o obj/load_config.o
gcc -Wall -Wextra -O3 -std=gnu11 -Isrc -c src/utilities/csv_writer.c -o obj/csv_writer.o

# Compile Models
echo 'Compiling models...'
gcc -Wall -Wextra -O3 -std=gnu11 -Isrc -c src/02-C-serial/monte_carlo_serial.c -o obj/monte_carlo_serial.o
gcc -Wall -Wextra -O3 -std=gnu11 -fopenmp -Isrc -c src/03-openMP/monte_carlo_omp.c -o obj/monte_carlo_omp.o

echo 'Compiling CUDA model...'
# Note: Adding -I/host-cuda/include to find curand_kernel.h
nvcc -O3 -arch=sm_60 -Xcompiler -fPIC -Isrc -I/host-cuda/include -c src/05-GPU/monte_carlos_cuda.cu -o obj/monte_carlo_cuda.o

# Compile Main
echo 'Compiling main runner...'
gcc -Wall -Wextra -O3 -std=gnu11 -Isrc -c src/main_runner.c -o obj/main_runner.o

# Link
echo 'Linking...'
# Note: Adding -L/host-cuda/lib64 to find libcurand.so
nvcc -O3 -arch=sm_60 -Xcompiler -fPIC -o bin/monte_carlo \\
    obj/main_runner.o obj/monte_carlo_serial.o obj/monte_carlo_omp.o obj/monte_carlo_cuda.o \\
    obj/load_binary.o obj/load_config.o obj/csv_writer.o \\
    -L/host-cuda/lib64 -L/host-cuda/lib \\
    -lm -lgsl -lgslcblas -lyaml -lpthread -lcudart -lcurand -Xcompiler \"-fopenmp\"
"

if [ ! -f "bin/monte_carlo" ]; then
    echo "Error: Compilation failed - binary not found"
    exit 1
fi
echo "Compilation successful!"
echo ""

# STEP 3: Run Simulation (Container)
echo "=========================================="
echo "STEP 3: Running GPU Simulation"
echo "=========================================="
# We run the simulation inside the container.
# Explicitly construct LD_LIBRARY_PATH to include GSL, YAML, and Host CUDA
$WRAP "$IMG" bash -c "
  export LD_LIBRARY_PATH=${EBROOTGSL}/lib:${EBROOTGSL}/lib64:${EBROOTLIBYAML}/lib:${EBROOTLIBYAML}/lib64:/host-cuda/lib64:/host-cuda/lib:\$LD_LIBRARY_PATH
  echo \"LD_LIBRARY_PATH=\$LD_LIBRARY_PATH\"
  ./bin/monte_carlo
"

echo ""
echo "=========================================="
echo "GPU Simulation Complete!"
echo "Job finished at: $(date)"
echo "=========================================="

