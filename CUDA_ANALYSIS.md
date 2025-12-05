# CUDA Monte Carlo Implementation Analysis

## CUDA Kernel Launch Configuration

**Launch Parameters:**
- **Threads per Block:** 256
- **Number of Blocks:** `(M + 255) / 256` (ceiling division to ensure all M trials are covered)
- **Shared Memory per Block:** `256 × 2 × N × sizeof(double)` bytes
- **Total Threads Launched:** `num_blocks × 256` (may exceed M, guarded by `if (trial_idx >= M) return`)

**Launch Syntax:**
```cuda
monte_carlo_kernel<<<num_blocks, threads_per_block, shared_mem_size>>>(
    M, N, k, x, mu_d, L_d, S_values_d, params->random_seed
);
```

**Example:** For M=10,000 trials:
- Blocks: (10000 + 255) / 256 = 40 blocks
- Total threads: 40 × 256 = 10,240 threads
- Active threads: 10,000 (240 threads exit early due to guard)

---

## Memory Layout Diagram

```mermaid
graph TB
    subgraph Host["Host Memory (CPU)"]
        H_mu["mu vector<br/>N × double<br/>params->mu->data"]
        H_Sigma["Sigma matrix<br/>N×N × double<br/>params->Sigma"]
        H_L["L_host<br/>N×N × double<br/>Row-major Cholesky"]
        H_results["result->S_values<br/>M × int<br/>Final results"]
    end

    subgraph Device["Device Memory (GPU Global)"]
        D_mu["mu_d<br/>N × double<br/>Mean returns"]
        D_L["L_d<br/>N×N × double<br/>Row-major Cholesky"]
        D_results["S_values_d<br/>M × int<br/>Crash counts per trial"]
    end

    subgraph Shared["Shared Memory (Per Block)"]
        S_mem["shared_mem[]<br/>256 × 2×N × double<br/>Per-block allocation"]
        S_Z["Z arrays<br/>256 × N × double<br/>Standard normal variates"]
        S_R["R arrays<br/>256 × N × double<br/>Correlated returns"]
    end

    subgraph Registers["Thread Registers"]
        R_rng["curandState_t rng_state<br/>Per-thread RNG state"]
        R_S["int S<br/>Crash count per trial"]
        R_idx["int trial_idx<br/>Thread ID"]
    end

    H_mu -->|cudaMemcpy<br/>Host→Device| D_mu
    H_L -->|cudaMemcpy<br/>Host→Device| D_L
    D_results -->|cudaMemcpy<br/>Device→Host| H_results
    
    S_mem -.->|Partitioned| S_Z
    S_mem -.->|Partitioned| S_R
    
    D_mu -->|Read-only| Registers
    D_L -->|Read-only| Registers
    Registers -->|Write| D_results
    
    style Host fill:#e1f5ff
    style Device fill:#fff4e1
    style Shared fill:#ffe1f5
    style Registers fill:#e1ffe1
```

**Memory Transfer Flow:**
1. **Host → Device:** `mu` (N doubles), `L` (N×N doubles)
2. **Kernel Execution:** Uses shared memory for Z and R arrays, reads from global memory (mu_d, L_d)
3. **Device → Host:** `S_values` (M integers)

---

## GPU Kernel Design Report

The CUDA implementation employs a straightforward parallelization strategy where each thread independently executes one complete Monte Carlo trial. The kernel performs five sequential steps: generating standard normal variates, applying Cholesky transformation for correlation, adding mean returns, counting crashes, and storing results. This design maximizes parallelism by eliminating inter-thread dependencies.

**Thread Hierarchy:** The kernel uses a one-dimensional grid with 256 threads per block, with blocks calculated as `(M + 255) / 256`. Each thread computes its global index via `blockIdx.x * blockDim.x + threadIdx.x`, directly mapping to the trial index for load-balanced execution.

**Memory Management:** Input data (mean vector μ and Cholesky matrix L) are copied to device global memory using `cudaMemcpy` before kernel launch. The Cholesky decomposition is computed on the host using GSL and converted from column-major to row-major for efficient GPU access. Dynamic shared memory is allocated per block for thread-local Z and R arrays, reducing global memory pressure.

**cuRAND:** Each thread maintains an independent cuRAND state initialized with `curand_init(seed, trial_idx, 0, &rng_state)`, ensuring statistical independence across threads. Threads generate N standard normal variates using `curand_normal_double()`, enabling massive parallelization without host-device communication overhead.

**Atomic Operations:** This implementation does not use atomic operations. Each thread writes directly to its unique position `S_values_d[trial_idx]`, eliminating write conflicts. The final reduction (counting extreme events where S ≥ k) is performed on the host CPU after results are transferred back, avoiding GPU-side atomic contention and simplifying the kernel design.

