# Hybrid MPI + OpenMP Parallel Implementation

This is a distributed-memory parallel implementation using hybrid MPI and OpenMP threading. The MPI+OpenMP implementation distributes work across multiple compute nodes (via MPI) while exploiting shared-memory parallelism within each node (via OpenMP threads).

## Overview

The hybrid MPI+OpenMP version builds upon the optimized OpenMP approach by adding a distributed-memory layer that enables scaling beyond the limits of a single compute node.

The previous OpenMP-only versions were constrained by the number of cores available on a single machine. The hybrid version combines the best of both worlds: MPI processes distribute Monte Carlo trials across multiple nodes, while OpenMP threads within each process exploit the shared-memory parallelism of multi-core nodes.

## Data Structures

**MonteCarloParams** - Input parameters including N assets, k crash threshold, covariance matrix Sigma, mean returns mu, threshold x, M trials, and random seed for deterministic runs for a fixed MPI/OpenMP configuration.

**MonteCarloResult** - Output results containing probability estimate P_hat, count of extreme events, S_values array storing crashes per trial, standard error, 95% confidence interval bounds, and maximum kernel timing measurements across MPI processes.

**MPIOpenMPModelState** - Internal state maintaining the shared Cholesky decomposition L matrix (replicated across processes), an array of random number generators (one per OpenMP thread per MPI process), MPI rank and size information, and thread count for hierarchical resource management.

## Core Functions

**mpi_openmp_init()** - Distributed initialization that computes the Cholesky decomposition once per MPI process and initializes dedicated random number generators for each OpenMP thread within that process. Each generator receives a deterministic seed of the form params->random_seed + mpi_rank * 1000 + thread_id, which is designed to give distinct random streams per thread and rank and enables reproducible runs for a fixed MPI/OpenMP configuration (same ranks, threads, and seed).

**mpi_openmp_run_trial()** - Optimized trial execution using thread-local workspace allocated per thread within each MPI process. Generates N independent standard normals from the thread's dedicated RNG, transforms them to correlated returns using R = mu + L*Z, and counts crashes. Each trial executes completely independently across the distributed system.

**mpi_openmp_simulate()** - Main distributed simulation driver that coordinates work across MPI processes and OpenMP threads. Each MPI process calculates its subset of trials, executes them using the optimized OpenMP parallel region, then participates in MPI reductions to aggregate results. Includes kernel timing measurements and minimal communication overhead.

**mpi_openmp_cleanup_state()** - Distributed cleanup that releases memory for all per-thread random number generators and the shared Cholesky matrix on each MPI process, ensuring there are no leaks in the model’s internal state.

## Advanced Distributed Random Number Generation Strategy

Each thread in the distributed system receives a deterministic seed of the form
base_seed + mpi_rank * 1000 + thread_id.
This ensures reproducible seeding for a fixed MPI/OpenMP configuration and provides reasonably independent streams for all (rank, thread) pairs without requiring coordination or communication. Because every thread owns its own RNG instance, no synchronization is needed during random number generation.

## Optimization Strategy

The hybrid MPI+OpenMP implementation employs a multi-layered optimization strategy for distributed parallel efficiency:

**Inter-Node Optimization (MPI Layer):**

- Well-balanced load: Monte Carlo trials distributed across MPI processes using a block + remainder scheme (ranks may differ by at most one trial).
- Minimal communication: Three collective operations per simulation (reduce count, gather S_values, reduce max kernel time)
- Replicated data strategy: Small problem data (Sigma, mu) replicated across processes to eliminate communication during computation
- Non-blocking MPI operations: Uses MPI_Ireduce and MPI_Igatherv for efficient aggregation

**Intra-Node Optimization (OpenMP Layer):**

- Per-thread RNGs (no critical sections): Each OpenMP thread within each MPI process owns an independent gsl_rng instance, enabling fully parallel random number generation within nodes
- Thread-local workspace allocation: Z and R vectors allocated per thread within the parallel region, eliminating repeated allocation overhead
- Static scheduling: Trials distributed to threads in contiguous blocks for optimal cache locality within each node
