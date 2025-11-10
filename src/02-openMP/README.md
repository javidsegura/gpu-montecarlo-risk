# OpenMP Parallel Implementation

This is the shared-memory parallel implementation using OpenMP threading. The implementation parallelizes Monte Carlo trials across multiple CPU cores while maintaining statistical correctness.

## Overview

The OpenMP version distributes Monte Carlo trials across multiple threads running concurrently on different CPU cores. Each thread maintains its own random number generator to avoid synchronization overhead, and thread-local workspace vectors are allocated once per thread rather than per trial to minimize allocation costs.

## Data Structures

**MonteCarloParams** - Input parameters including N assets, k crash threshold, covariance matrix Sigma, mean returns mu, threshold x, and M trials.

**MonteCarloResult** - Output results containing probability estimate P_hat, count of extreme events, S_values array storing crashes per trial, standard error, and 95% confidence interval bounds.

**OpenMPModelState** - Internal state maintaining the shared Cholesky decomposition L matrix, an array of random number generators (one per thread), and the thread count.

## Core Functions

**openmp_init()** - Preprocessing that computes the Cholesky decomposition once and initializes one random number generator per OpenMP thread. Each generator is seeded with 42, though this does not guarantee identical results to the serial version due to non-deterministic thread scheduling.

**openmp_run_trial()** - Executes a single Monte Carlo trial using thread-local workspace. Generates N independent standard normals from the thread's dedicated RNG, transforms them to correlated returns using R = mu + L*Z, and counts crashes.

**openmp_simulate()** - Main simulation driver that creates a parallel region where each thread allocates workspace vectors once, then executes its assigned subset of trials using static scheduling. The count of extreme events is accumulated using OpenMP reduction.

**openmp_cleanup_state()** - Releases memory for all thread-local random number generators and the shared Cholesky matrix.


## Random Number Generation and Reproducibility

The OpenMP implementation uses per-thread random number generators to avoid lock contention and synchronization overhead. Each thread's RNG is initialized with seed 42, but this does not guarantee bit-for-bit identical results compared to the serial version or across different runs with the same number of threads.

The reason is that OpenMP threads execute trials in a non-deterministic order depending on OS scheduling decisions. While thread 0 might execute trials {0, 100, 200} and thread 1 might execute {1, 101, 201} in one run, the assignment could differ in subsequent runs even with the same seed and thread count. This means the random number sequence gets mapped to different trials across runs.

Despite this non-determinism in exact trial outcomes, the statistical properties remain valid. The estimated probability P_hat and confidence intervals are correct, and results should fall within the expected statistical variation (typically within the standard error of the serial version). For instance, if the serial version produces P_hat = 0.4974 with standard error 0.0016, the OpenMP version might produce 0.4984, which differs slightly but remains within acceptable statistical bounds.

For applications requiring exact reproducibility across parallel and serial runs, alternative approaches include pre-generating all random numbers serially before parallelization or using specialized parallel random number generators with leap-frog or sequence-splitting techniques. However, these methods introduce additional complexity and potential performance overhead that are unnecessary for Monte Carlo simulations where statistical correctness is the primary concern.

## Optimization Strategy

The implementation employs several optimizations for parallel efficiency:

- Thread-local workspace allocation: Z and R vectors are allocated once per thread rather than per trial, eliminating repeated allocation overhead.
- Static scheduling: Trials are distributed to threads in contiguous blocks for better cache locality and reduced scheduling overhead. Static scheduling is appropriate here because all trials have uniform computational cost.
- Per-thread RNGs: Each thread maintains its own random number generator to eliminate lock contention that would occur with a shared generator.
- Reduction clause: The count of extreme events is accumulated in parallel using OpenMP's built-in reduction mechanism.
- Shared Cholesky decomposition: The L matrix is computed once and shared read-only across all threads, avoiding redundant computation.
