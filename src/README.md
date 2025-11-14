# CLUSTER
Because of memory issues clone the repo in the scratch folder.

```bash
git clone https://github.com/javidsegura/gpu-montecarlo-risk
cd gpu-montecarlo-risk
```

RUN:
1. Create a makefile to compile the code in the cluster.


2. Run:
```bash
sbatch slurm_script.sh
```

# LOCAL

## Building the Project

The project uses a Makefile located in the root directory. All build artifacts are placed in `build/` (object files) and `bin/` (executables).

### Compilation

Build the entire project with a single command:

```bash
make
```

This compiles all implementations (serial and OpenMP) into one executable at `bin/monte_carlo`.

### Clean Build Artifacts

Remove all compiled files:

```bash
make clean
```

Clean and rebuild everything:

```bash
make rebuild
```

## Running Simulations


### Examples
```

Run serial + OpenMP with 8 threads:

```bash
OMP_NUM_THREADS=8 ./bin/monte_carlo 
```

### Timing Execution

Use the `time` command to measure performance:

```bash
time ./bin/monte_carlo 
```

## Dependencies

The project requires the following libraries:

- **GSL (GNU Scientific Library)**: Provides linear algebra operations, random number generation, and statistical functions.
- **OpenMP**: Included with GCC for parallel execution on shared-memory systems.

### Installing Dependencies on Ubuntu/Debian

```bash
sudo apt-get install libgsl-dev
```

### Installing Dependencies on macOS

```bash
brew install gsl
```

## Implementation Details

Each implementation follows a common interface defined in `model_interface.h`, which specifies:

- **MonteCarloParams**: Input parameters (N assets, k threshold, M trials, covariance matrix, etc.)
- **MonteCarloResult**: Output results (probability estimate, confidence intervals, trial data)
- **ModelFunctions**: Function pointers for initialization, execution, and cleanup

All implementations compute the same quantity: the probability that at least k out of N assets experience returns below a threshold -x in a single time period. The algorithm generates correlated asset returns using Cholesky decomposition and counts extreme events across M independent Monte Carlo trials.
