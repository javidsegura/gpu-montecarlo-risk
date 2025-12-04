#!/bin/bash
#SBATCH --job-name=montecarlo_mpi
#SBATCH --partition=gpu-node
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --time=00:15:00
#SBATCH --output=slurm-mpi-%j.out
#SBATCH --error=slurm-mpi-%j.err

set -e

echo "=========================================="
echo "Monte Carlo Risk Analysis (MPI-CUDA)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NODELIST"
echo "Number of tasks: $SLURM_NTASKS"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "Date: $(date)"
echo "=========================================="

# Load host modules for Preprocessing and Path Discovery
module purge
module load StdEnv/2023
module load python/3.11.5
module load gsl/2.7
module load libyaml/0.2.5
module load hwloc/2.9.1
module load openmpi/4.1.5

echo ""
echo "=== Loaded Modules (Host) ==="
module list
echo "============================="
echo ""

# Define paths
WRAP="/project/dlstack/tools/appt-gpu"
IMG="/project/dlstack/containers/tensorflow-2.16.1-gpu.sif"

# Host CUDA Path (for headers/libs missing in container)
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

# Check for MPI
if [ -z "$EBROOTOPENMPI" ]; then
    echo "Error: EBROOTOPENMPI not set. OpenMPI module might not be loaded correctly."
    exit 1
fi

# Get hwloc path (MPI dependency)
if [ -z "$EBROOTHWLOC" ]; then
    echo "Warning: EBROOTHWLOC not set, trying to find hwloc..."
    EBROOTHWLOC=$(module show hwloc 2>&1 | grep "EBROOTHWLOC" | sed 's/.*"\(.*\)".*/\1/' | head -1)
    if [ -z "$EBROOTHWLOC" ]; then
        echo "Error: Could not find hwloc. Please load hwloc module."
        exit 1
    fi
fi

echo "GSL Path: $EBROOTGSL"
echo "LibYAML Path: $EBROOTLIBYAML"
echo "Host CUDA Path: $CUDA_HOST"
echo "OpenMPI Path: $EBROOTOPENMPI"
echo "HWLOC Path: $EBROOTHWLOC"

# Setup Container Bindings
# We bind the project directory (pwd), GSL, LibYAML, Host CUDA, MPI, hwloc, and gentoo system libs
# Gentoo libs contain libevent, libpciaccess, libze_loader needed by MPI/hwloc
GENTOO_LIBS="/cvmfs/soft.computecanada.ca/gentoo/2023/x86-64-v3/usr/lib64"
export APPTAINER_BIND="$(pwd),${EBROOTGSL},${EBROOTLIBYAML},${EBROOTOPENMPI}:/host-mpi,${EBROOTHWLOC}:/host-hwloc,${GENTOO_LIBS}:/host-gentoo-libs,${CUDA_HOST}:/host-cuda"

# Setup Environment Variables for Compilation inside Container
# GCC, NVCC, and MPICC will look at these for headers and libraries
GENTOO_LIBS="/cvmfs/soft.computecanada.ca/gentoo/2023/x86-64-v3/usr/lib64"
export CPATH="${EBROOTGSL}/include:${EBROOTLIBYAML}/include:${EBROOTOPENMPI}/include:${EBROOTHWLOC}/include:$CPATH"
export LIBRARY_PATH="${EBROOTGSL}/lib:${EBROOTLIBYAML}/lib:${EBROOTOPENMPI}/lib:${EBROOTHWLOC}/lib:${GENTOO_LIBS}:$LIBRARY_PATH"
export LD_LIBRARY_PATH="${EBROOTGSL}/lib:${EBROOTLIBYAML}/lib:${EBROOTOPENMPI}/lib:${EBROOTHWLOC}/lib:${GENTOO_LIBS}:$LD_LIBRARY_PATH"
export PATH="${EBROOTOPENMPI}/bin:$PATH"

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
pip install --no-cache-dir --upgrade pip > /dev/null 2>&1
pip install --no-cache-dir -r minimal_req.txt > /dev/null 2>&1

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

echo 'Compiling CUDA model with MPI support...'
# Note: Adding -I/host-cuda/include to find curand_kernel.h
# Compile with USE_MPI flag to enable MPI-CUDA functionality
# MPI is bound to /host-mpi in container
nvcc -O3 -arch=sm_60 -Xcompiler -fPIC -DUSE_MPI -Isrc -I/host-cuda/include -I/host-mpi/include -c src/05-GPU/monte_carlos_cuda.cu -o obj/monte_carlo_cuda.o

# Compile Main
echo 'Compiling main runner...'
gcc -Wall -Wextra -O3 -std=gnu11 -Isrc -c src/main_runner.c -o obj/main_runner.o

# Link
echo 'Linking...'
# Use nvcc for linking (required for CUDA objects)
# Get MPI linker flags from mpicc to ensure all dependencies are included
MPI_LIBS=\$(mpicc --showme:link 2>/dev/null || echo \"-lmpi\")
# Use nvcc but pass MPI libraries via -Xlinker
# Add gentoo system libraries for libevent, libpciaccess, libze_loader (needed by MPI/hwloc)
# Note: hwloc's PCI/ZE features are optional - link against them if available
nvcc -O3 -arch=sm_60 -Xcompiler -fPIC -o bin/monte_carlo \\
    obj/main_runner.o obj/monte_carlo_serial.o obj/monte_carlo_omp.o \\
    obj/monte_carlo_cuda.o \\
    obj/load_binary.o obj/load_config.o obj/csv_writer.o \\
    -L/host-cuda/lib64 -L/host-cuda/lib -L/host-mpi/lib -L/host-hwloc/lib -L/host-gentoo-libs \\
    -lm -lgsl -lgslcblas -lyaml -lpthread -lcudart -lcurand \\
    -levent_core -levent_pthreads -lpciaccess -lze_loader \\
    -Xlinker \"\$MPI_LIBS\" -Xcompiler \"-fopenmp\" \\
    -Xlinker \"-rpath=/host-mpi/lib:/host-hwloc/lib:/host-gentoo-libs:/host-cuda/lib64:/host-cuda/lib\"
"

if [ ! -f "bin/monte_carlo" ]; then
    echo "Error: Compilation failed - binary not found"
    exit 1
fi
echo "Compilation successful!"
echo ""

# STEP 3: Run Simulation (Container with MPI)
echo "=========================================="
echo "STEP 3: Running MPI-CUDA Simulation"
echo "=========================================="
# Run with mpirun inside the container
# Explicitly construct LD_LIBRARY_PATH to include GSL, YAML, Host CUDA, and MPI
$WRAP "$IMG" bash -c "
  export LD_LIBRARY_PATH=${EBROOTGSL}/lib:${EBROOTGSL}/lib64:${EBROOTLIBYAML}/lib:${EBROOTLIBYAML}/lib64:/host-mpi/lib:/host-hwloc/lib:/host-gentoo-libs:/host-cuda/lib64:/host-cuda/lib:\$LD_LIBRARY_PATH
  export PATH=/host-mpi/bin:\$PATH
  echo \"LD_LIBRARY_PATH=\$LD_LIBRARY_PATH\"
  echo \"Running with \$SLURM_NTASKS MPI ranks...\"
  # Use config_mpi.yaml if it exists, otherwise use config.yaml
  if [ -f config_mpi.yaml ]; then
    # Temporarily rename config files so MPI version uses the right one
    mv config.yaml config.yaml.bak 2>/dev/null || true
    mv config_mpi.yaml config.yaml
  fi
  # Suppress stdin input prompt by piping empty input
  echo \"\" | mpirun -np \$SLURM_NTASKS --bind-to none ./bin/monte_carlo
  # Restore original config
  if [ -f config.yaml.bak ]; then
    mv config.yaml config_mpi.yaml
    mv config.yaml.bak config.yaml
  fi
"

echo ""
echo "=========================================="
echo "MPI-CUDA Simulation Complete!"
echo "Job finished at: $(date)"
echo "=========================================="

