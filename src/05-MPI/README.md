# Hybrid MPI + OpenMP Parallel Implementation

This is a distributed-memory parallel implementation using hybrid MPI and OpenMP threading. The MPI+OpenMP implementation distributes work across multiple compute nodes (via MPI) while exploiting shared-memory parallelism within each node (via OpenMP threads).

## Overview

The hybrid MPI+OpenMP version builds upon the optimized OpenMP approach by adding a distributed-memory layer that enables scaling beyond the limits of a single compute node.

Previous OpenMP-only versions were constrained by the number of cores available on a single machine. The hybrid version combines the best of both worlds: MPI processes distribute Monte Carlo trials across multiple nodes, while OpenMP threads within each process exploit shared-memory parallelism inside each node.

## Data Structures

**MonteCarloParams** – Input parameters including N assets, k crash threshold, covariance matrix Sigma, mean returns mu, threshold x, total trials M, and the random seed used to ensure deterministic behavior for any *fixed* MPI/OpenMP configuration.

**MonteCarloResult** – Output results containing estimated probability P_hat, total extreme event count, standard error, 95% confidence interval bounds, and the maximum kernel time measured across MPI ranks. Note: S_values is set to NULL in the MPI version to reduce memory usage and drastically reduce communication cost on distributed environments.

**MPIOpenMPModelState** – Internal model state containing the shared Cholesky decomposition matrix L (replicated on each MPI process), an array of per-thread random number generators (one gsl_rng per OpenMP thread), MPI rank and size information, and thread-count metadata used for hierarchical parallelism management.

## Core Functions

**mpi_openmp_init()** – Distributed initialization routine executed once per MPI rank. It computes the Cholesky decomposition of Sigma, allocates one RNG per OpenMP thread, and seeds each RNG deterministically using:

```
params->random_seed + mpi_rank * 1000 + thread_id
```

This guarantees independent random streams for each (rank, thread) pair and produces reproducible simulation results for any fixed MPI+OpenMP configuration.

---

**mpi_openmp_run_trial()** – Executes one Monte Carlo trial using thread-local workspace. Each thread generates N standard normal samples, computes correlated returns via:

```
R = mu + L * Z
```

and counts the number of crashes. No synchronization is required between threads or ranks: each trial is fully independent.

---

**mpi_openmp_simulate()** – Main distributed driver coordinating work across MPI and OpenMP:

- Uses a block + remainder workload distribution so that ranks differ by at most one trial.
- Performs all trial computations in parallel using OpenMP threads within each rank.
- Aggregates global results using **non-blocking MPI_Ireduce** calls for:
  - total count of extreme events
  - maximum kernel time across ranks

This avoids collecting per-trial data and drastically reduces communication overhead, allowing the model to scale efficiently across multiple nodes.

---

**mpi_openmp_cleanup_state()** – Releases all internal model resources, including all per-thread RNGs and the replicated Cholesky matrix, ensuring each MPI rank frees its memory correctly.

## Design Decision: Why S_values Are Not Collected

The MPI implementation uses a **streaming computation model** that computes but does not store per-trial crash counts (S_values). This is a deliberate design choice based on the mathematical requirements of Monte Carlo estimation.

### What Happens in Each Trial

For every trial j, the code:

1. **Computes S** (number of crashes): This value is calculated by `mpi_openmp_run_trial()` and returned as a temporary variable
2. **Tests the threshold**: Checks if `S >= k` (extreme event condition)
3. **Updates the counter**: If the condition is true, increments `local_count`
4. **Discards S**: The value of S is immediately overwritten in the next iteration

### Comparison with OpenMP Version

**OpenMP Implementation:**
```c
int S = 0;
openmp_run_trial(..., &S);
result->S_values[j] = S;    // ← STORES S in array
if (S >= k) {
    count++;
}
```

**MPI Implementation:**
```c
int S = 0;
mpi_openmp_run_trial(..., &S);
// No storage - S exists only temporarily
if (S >= k) {
    local_count++;          // ← ONLY accumulates count
}
```

### Why This Is Mathematically Sufficient

The Monte Carlo probability estimate is computed as:

```
P_hat = (number of trials where S >= k) / M
```

The individual S values are **not required** for this calculation. Only the **total count** of extreme events matters. Statistical measures (standard error, confidence intervals) are also derived solely from this count:

```
std_error = sqrt(P_hat * (1 - P_hat) / M)
CI = P_hat ± 1.96 * std_error
```

### Benefits of Not Storing S_values

1. **Memory efficiency**: Storage is O(N) instead of O(M). For M = 10^9 trials, this saves ~4 GB of RAM
2. **Communication efficiency**: Only two scalar reductions instead of gathering M integers across all ranks
3. **Scalability**: Enables simulations with billions of trials on distributed systems
4. **Streaming computation**: Each trial's memory is reused, improving cache performance

### When S_values Would Be Useful

The S_values array is valuable for:
- Debugging and validation during development
- Post-processing analysis (e.g., histograms of crash counts)
- Detailed statistical diagnostics

However, for production Monte Carlo estimation on distributed systems, storing per-trial data is unnecessary and counterproductive to scalability.

## Advanced Distributed Random Number Generation Strategy

Each thread in the system receives a deterministic seed of the form:

```
base_seed + mpi_rank * 1000 + thread_id
```

This strategy provides:

- Reproducible random streams across the entire distributed job
- Full independence between all (rank, thread) pairs
- Zero communication or synchronization for RNG management
- Scalable generation suitable for large HPC systems

The spacing of 1000 between ranks assumes fewer than 1000 threads per process, which is far above typical HPC configurations. Because every thread owns its own RNG instance, **no critical sections or locks are required**, ensuring maximal OpenMP throughput.

## Optimization Strategy

The hybrid MPI + OpenMP implementation applies complementary optimization methods at both distributed-memory and shared-memory levels.

### Inter-Node Optimization (MPI Layer)

- **Balanced load distribution:** Trials are evenly allocated using block + remainder.
- **Minimal communication:** Only two collective reductions are performed per simulation.
- **Replicated-data model:** Sigma and mu are broadcast implicitly by duplication, avoiding communication during compute.
- **Memory-efficient aggregation:** The MPI version does **not** gather per-trial results (S_values), drastically reducing memory and bandwidth requirements.
- **Non-blocking collectives:** MPI_Ireduce + MPI_Waitall minimize global synchronization time.

### Intra-Node Optimization (OpenMP Layer)

- **Per-thread RNGs:** Eliminates thread contention and synchronization.
- **Thread-local workspace:** Vectors Z and R are allocated once per thread, avoiding repeat allocations.
- **Static scheduling:** Ensures cache-friendly access patterns and balanced thread workloads.
- **OpenMP reduction:** Efficiently accumulates per-thread extreme event counts.

### Scalability Benefits

- **No memory explosion:** Avoiding S_values collection keeps memory usage proportional to O(N), not O(M).
- **Low communication volume:** Only two scalar reductions regardless of M.
- **Massive parallel potential:** Supports hundreds of MPI ranks × dozens of threads per rank without communication bottlenecks.
- **Excellent for large M:** Designed specifically for multi-node clusters executing millions or billions of trials.
