# Optimized OpenMP Parallel Implementation

This is an advanced shared-memory parallel implementation using OpenMP threading with optimizations. The optimized OpenMP implementation delivers substantial performance improvements over the original design by eliminating unnecessary synchronization inside the Monte Carlo simulation kernel.

## Overview
The optimized OpenMP version builds upon the standard OpenMP approach. 

The previous version relied on a shared GSL random number generator protected by a critical section, which forced all threads to serialize during random-number generation. This effectively eliminated parallelism in the hottest part of the computation and prevented meaningful scaling.

The optimized version assigns each OpenMP thread its own deterministic, independent random number generator, removing the need for locks and allowing threads to progress without blocking each other.


## Data Structures
**MonteCarloParams** - Input parameters including N assets, k crash threshold, covariance matrix Sigma, mean returns mu, threshold x, M trials, and random seed for reproducible results.

**MonteCarloResult** - Output results containing probability estimate P_hat, count of extreme events, S_values array storing crashes per trial, standard error, 95% confidence interval bounds, and kernel timing measurements.

**OpenMPModelState** - Internal state maintaining the shared Cholesky decomposition L matrix, an array of random number generators (one per thread), and the thread count for resource management.

## Core Functions
**openmp_init()** - Advanced preprocessing that computes the Cholesky decomposition once and initializes a dedicated random number generator for each OpenMP thread. Each generator receives a deterministic seed based on params->random_seed + thread_id, ensuring reproducible results across runs with the same thread count and scheduling.

**openmp_run_trial()** - Highly optimized trial execution using pre-allocated thread-local workspace. Generates N independent standard normals from the thread's dedicated RNG without any synchronization, transforms them to correlated returns using efficient BLAS operations (R = mu + L*Z), and counts crashes through vectorized comparison.

**openmp_opt_simulate()** - Main simulation driver implementing a zero-overhead parallel region. Each thread allocates workspace vectors once at the beginning of the parallel section, then executes its assigned trials using static scheduling for optimal cache locality. Includes precise kernel timing measurements and uses OpenMP reduction for lock-free accumulation of results.

**openmp_cleanup_state()** - Comprehensive cleanup that releases memory for all thread-local random number generators and the shared Cholesky matrix, ensuring no memory leaks.

## Advanced Random Number Generation Strategy
The optimized implementation uses a per-thread RNG strategy that achieves both high performance and deterministic reproducibility:

**Deterministic Seeding Strategy**
Each thread's RNG receives a unique, deterministic seed calculated as base_seed + thread_id. This approach ensures:

- Reproducible results across runs with the same parameters and thread count
- Independent random streams between threads with no correlation
- Scalable performance with zero synchronization overhead

**Reproducibility Guarantees**
This optimized implementation provides strong reproducibility guarantees:

- Same input parameters + same thread count = identical results
- Static scheduling ensures deterministic work distribution
- Deterministic seeding eliminates randomness in thread startup order
- The combination of static scheduling with deterministic per-thread seeding means that trial j will always be executed by the same thread using the same random number sequence, regardless of OS scheduling variations.

## Performance Benefits
By eliminating the critical section in the basic OpenMP version, this approach achieves:

Cleaner CPU utilization, reduced overhead, and more efficient use of compute resources.

See Report for details.

## Optimization Strategy 
The optimized OpenMP implementation employs the following optimizations for parallel efficiency:

- Per-thread RNGs (no critical section):
Each thread owns an independent gsl_rng * instance, so random number generation is fully parallel. This removes the severe bottleneck of threads queueing on a critical section to access a single shared RNG.
- Thread-local workspace allocation: Z and R vectors are allocated once per thread rather than per trial, eliminating repeated allocation overhead.
- Static scheduling: Trials are distributed to threads in contiguous blocks for better cache locality and reduced scheduling overhead. Static scheduling is appropriate here because all trials have uniform computational cost.
- Per-thread RNGs: Each thread maintains its own random number generator to eliminate lock contention that would occur with a shared generator.
- Reduction clause: The count of extreme events is accumulated in parallel using OpenMP's built-in reduction mechanism.
- Shared Cholesky decomposition: The L matrix is computed once and shared read-only across all threads, avoiding redundant computation.
