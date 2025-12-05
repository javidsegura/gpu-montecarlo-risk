#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>

extern "C" {
#include "../model_interface.h"
}

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            return -1; \
        } \
    } while (0)

__global__ void monte_carlo_kernel(int M, int N, int k, double x, 
                                   const double *d_mu, const double *d_L, 
                                   unsigned long seed, int *d_S_values, 
                                   int *d_count_reduction) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M) return;

    curandState state;
    curand_init(seed, idx, 0, &state);

    // Thread-local arrays (registers/local memory)
    const int MAX_N = 64; 
    double Z[MAX_N];
    double R[MAX_N];
    
    // Ensure N is within bounds
    if (N > MAX_N) {
        // Should be checked on host, but failsafe here
        return; 
    }
    
    // Generate Z (Standard Normal)
    for (int i = 0; i < N; i++) {
        Z[i] = curand_normal_double(&state);
    }

    // R = mu + L * Z
    // L is lower triangular
    for (int i = 0; i < N; i++) {
        double sum = 0.0;
        for (int j = 0; j <= i; j++) {
            sum += d_L[i * N + j] * Z[j];
        }
        R[i] = d_mu[i] + sum;
    }

    // Count crashes
    int S = 0;
    for (int i = 0; i < N; i++) {
        if (R[i] < -x) {
            S++;
        }
    }

    d_S_values[idx] = S;
    
    if (S >= k) {
        atomicAdd(d_count_reduction, 1);
    }
}

// Host wrapper
extern "C" int cuda_simulate(MonteCarloParams *params, MonteCarloResult *result) {
    printf("Starting CUDA Monte Carlo simulation with M = %d trials...\n", params->M);
    
    if (params->N > 64) {
        fprintf(stderr, "Error: N > 64 not supported in this CUDA kernel implementation\n");
        return -1;
    }

    // 1. Prepare Cholesky on Host
    gsl_matrix *L = gsl_matrix_alloc(params->N, params->N);
    if (!L) return -1;
    
    gsl_matrix_memcpy(L, params->Sigma);
    int status = gsl_linalg_cholesky_decomp1(L); // In-place
    if (status) {
        fprintf(stderr, "Error: Cholesky decomposition failed\n");
        gsl_matrix_free(L);
        return -1;
    }

    // Flat arrays for copy
    double *h_mu_flat = (double*)malloc(params->N * sizeof(double));
    for(int i=0; i<params->N; i++) h_mu_flat[i] = gsl_vector_get(params->mu, i);

    double *h_L_flat = (double*)malloc(params->N * params->N * sizeof(double));
    // Zero out upper triangle just in case, though kernel uses loop limits
    for(int i=0; i<params->N; i++) {
        for(int j=0; j<params->N; j++) {
             if (j <= i)
                h_L_flat[i*params->N + j] = gsl_matrix_get(L, i, j);
             else
                h_L_flat[i*params->N + j] = 0.0;
        }
    }

    // 2. Allocate Device Memory
    double *d_mu, *d_L;
    int *d_S_values, *d_count;
    
    if (cudaMalloc(&d_mu, params->N * sizeof(double)) != cudaSuccess) return -1;
    if (cudaMalloc(&d_L, params->N * params->N * sizeof(double)) != cudaSuccess) return -1;
    if (cudaMalloc(&d_S_values, params->M * sizeof(int)) != cudaSuccess) return -1;
    if (cudaMalloc(&d_count, sizeof(int)) != cudaSuccess) return -1;

    // 3. Copy Data
    cudaMemcpy(d_mu, h_mu_flat, params->N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_L, h_L_flat, params->N * params->N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(d_count, 0, sizeof(int));

    // 4. Launch Kernel
    int blockSize = 256;
    int numBlocks = (params->M + blockSize - 1) / blockSize;
    monte_carlo_kernel<<<numBlocks, blockSize>>>(params->M, params->N, params->k, params->x, 
                                                 d_mu, d_L, params->random_seed, 
                                                 d_S_values, d_count);
    
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
         fprintf(stderr, "CUDA Kernel Failed: %s\n", cudaGetErrorString(err));
         return -1;
    }

    // 5. Copy Results Back
    result->S_values = (int*)malloc(params->M * sizeof(int));
    cudaMemcpy(result->S_values, d_S_values, params->M * sizeof(int), cudaMemcpyDeviceToHost);
    
    int h_count;
    cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    result->count = h_count;

    // 6. Stats
    result->P_hat = (double)result->count / params->M;
    result->std_error = sqrt(result->P_hat * (1.0 - result->P_hat) / params->M);
    double margin = 1.96 * result->std_error;
    result->ci_lower = fmax(0.0, result->P_hat - margin);
    result->ci_upper = fmin(1.0, result->P_hat + margin);

    // Cleanup
    cudaFree(d_mu);
    cudaFree(d_L);
    cudaFree(d_S_values);
    cudaFree(d_count);
    free(h_mu_flat);
    free(h_L_flat);
    gsl_matrix_free(L);

    printf("CUDA Simulation complete\n");
    return 0;
}

extern "C" ModelFunctions get_cuda_model(void) {
    ModelFunctions model = {
        .name = "cuda",
        .run_model = cuda_simulate
    };
    return model;
}

