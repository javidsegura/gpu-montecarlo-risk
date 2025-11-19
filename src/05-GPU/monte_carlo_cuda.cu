// Monte Carlo simulation for financial crash probability - CUDA VERSION
// 
// PARALLELIZATION STRATEGY:
// - Each CUDA thread handles ONE Monte Carlo trial
// - Uses cuRAND for GPU-based random number generation
// - Matrix operations performed on GPU
// - Results copied back to host for compatibility with existing infrastructure

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_linalg.h>
#include "../model_interface.h"

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            return -1; \
        } \
    } while(0)

// CUDA kernel: Each thread runs ONE Monte Carlo trial
// trial_idx: index of this trial (0 to M-1)
// N: number of assets
// k: threshold for extreme events
// x: crash threshold
// mu_d: mean returns (device memory)
// L_d: Cholesky decomposition (device memory, row-major N x N)
// S_values_d: output array for crash counts (device memory)
// seed: random seed for cuRAND
__global__ void monte_carlo_kernel(int M, int N, int k, double x,
                                   double *mu_d, double *L_d,
                                   int *S_values_d, unsigned long seed) {
    // Global thread ID = trial index
    int trial_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (trial_idx >= M) return;  // Guard against extra threads

    // Initialize random number generator for this thread
    curandState_t rng_state;
    curand_init(seed, trial_idx, 0, &rng_state);

    // Allocate thread-local arrays for Z (standard normal) and R (returns)
    // For small N, use stack allocation; for large N would need dynamic shared memory
    extern __shared__ double shared_mem[];
    double *Z = &shared_mem[threadIdx.x * N * 2];
    double *R = &shared_mem[threadIdx.x * N * 2 + N];

    // Step 1: Generate N independent standard normal variates
    for (int i = 0; i < N; i++) {
        Z[i] = curand_normal_double(&rng_state);
    }

    // Step 2: Transform to correlated returns: R = L * Z
    // L is stored in row-major format
    for (int i = 0; i < N; i++) {
        R[i] = 0.0;
        for (int j = 0; j <= i; j++) {  // L is lower triangular
            R[i] += L_d[i * N + j] * Z[j];
        }
    }

    // Step 3: Add mean: R = R + mu
    for (int i = 0; i < N; i++) {
        R[i] += mu_d[i];
    }

    // Step 4: Count crashes (returns below -x threshold)
    int S = 0;
    for (int i = 0; i < N; i++) {
        if (R[i] < -x) {
            S++;
        }
    }

    // Step 5: Store result
    S_values_d[trial_idx] = S;
}

// Host function: Setup and launch CUDA kernel
static int cuda_simulate(MonteCarloParams *params, MonteCarloResult *result) {
    printf("Starting CUDA Monte Carlo simulation with M = %d trials...\n", params->M);
    printf("Parameters: N=%d, k=%d, x=%.2f%%\n", params->N, params->k, params->x * 100);

    int M = params->M;
    int N = params->N;
    int k = params->k;
    double x = params->x;

    // Query GPU properties
    int device = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("Using GPU: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);

    // STEP 1: Allocate host result array
    result->S_values = (int *)calloc(M, sizeof(int));
    if (!result->S_values) {
        fprintf(stderr, "Error: Failed to allocate host S_values array\n");
        return -1;
    }

    // STEP 2: Allocate device memory
    double *mu_d = NULL;
    double *L_d = NULL;
    int *S_values_d = NULL;

    CUDA_CHECK(cudaMalloc(&mu_d, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&L_d, N * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&S_values_d, M * sizeof(int)));

    // STEP 3: Prepare Cholesky decomposition on host
    // GSL uses column-major, we need row-major for efficient GPU access
    double *L_host = (double *)malloc(N * N * sizeof(double));
    if (!L_host) {
        fprintf(stderr, "Error: Failed to allocate host Cholesky matrix\n");
        cudaFree(mu_d);
        cudaFree(L_d);
        cudaFree(S_values_d);
        free(result->S_values);
        return -1;
    }

    // Compute Cholesky decomposition using GSL
    gsl_matrix *Sigma_copy = gsl_matrix_alloc(N, N);
    gsl_matrix_memcpy(Sigma_copy, params->Sigma);
    int status = gsl_linalg_cholesky_decomp1(Sigma_copy);
    if (status) {
        fprintf(stderr, "Error: Cholesky decomposition failed\n");
        gsl_matrix_free(Sigma_copy);
        free(L_host);
        cudaFree(mu_d);
        cudaFree(L_d);
        cudaFree(S_values_d);
        free(result->S_values);
        return -1;
    }

    // Convert from GSL column-major to row-major
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            L_host[i * N + j] = gsl_matrix_get(Sigma_copy, i, j);
        }
    }
    gsl_matrix_free(Sigma_copy);

    // STEP 4: Copy data to device
    CUDA_CHECK(cudaMemcpy(mu_d, params->mu->data, N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(L_d, L_host, N * N * sizeof(double), cudaMemcpyHostToDevice));
    free(L_host);

    // STEP 5: Configure kernel launch parameters
    int threads_per_block = 256;
    int num_blocks = (M + threads_per_block - 1) / threads_per_block;
    
    // Calculate shared memory size: each thread needs 2*N doubles (Z and R)
    size_t shared_mem_size = threads_per_block * 2 * N * sizeof(double);
    
    printf("Launching kernel with %d blocks × %d threads = %d threads\n",
           num_blocks, threads_per_block, num_blocks * threads_per_block);
    printf("Shared memory per block: %.2f KB\n", shared_mem_size / 1024.0);

    // Check if shared memory requirement is feasible
    if (shared_mem_size > prop.sharedMemPerBlock) {
        fprintf(stderr, "Error: Shared memory requirement (%zu bytes) exceeds limit (%zu bytes)\n",
                shared_mem_size, prop.sharedMemPerBlock);
        fprintf(stderr, "Consider reducing N or using global memory for large problems\n");
        cudaFree(mu_d);
        cudaFree(L_d);
        cudaFree(S_values_d);
        free(result->S_values);
        return -1;
    }

    // STEP 6: Launch kernel
    monte_carlo_kernel<<<num_blocks, threads_per_block, shared_mem_size>>>(
        M, N, k, x, mu_d, L_d, S_values_d, params->random_seed
    );

    // Check for kernel launch errors
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(launch_err));
        cudaFree(mu_d);
        cudaFree(L_d);
        cudaFree(S_values_d);
        free(result->S_values);
        return -1;
    }

    // Wait for kernel to complete
    CUDA_CHECK(cudaDeviceSynchronize());

    // STEP 7: Copy results back to host
    CUDA_CHECK(cudaMemcpy(result->S_values, S_values_d, M * sizeof(int), cudaMemcpyDeviceToHost));

    // STEP 8: Count extreme events on host
    result->count = 0;
    for (int j = 0; j < M; j++) {
        if (result->S_values[j] >= k) {
            result->count++;
        }
    }

    printf("Simulation complete\n");

    // STEP 9: Calculate final probability estimate
    result->P_hat = (double)result->count / M;

    // STEP 10: Calculate accuracy metrics (standard error and 95% CI)
    result->std_error = sqrt(result->P_hat * (1.0 - result->P_hat) / M);
    double margin = 1.96 * result->std_error;
    result->ci_lower = fmax(0.0, result->P_hat - margin);
    result->ci_upper = fmin(1.0, result->P_hat + margin);

    // STEP 11: Cleanup device memory
    cudaFree(mu_d);
    cudaFree(L_d);
    cudaFree(S_values_d);

    return 0;
}

// External C linkage for main_runner
extern "C" {
    ModelFunctions get_cuda_model(void) {
        ModelFunctions model = {
            .name = "cuda",
            .run_model = cuda_simulate
        };
        return model;
    }
}

