#include <stdio.h>
#include <stdlib.h>
#include <math.h>
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

// Constant memory for small matrices (N <= 64) - cached, fast access
__constant__ double c_mu[64];
__constant__ double c_L[64 * 64];

// Optimized kernel with constant memory, faster RNG, and manual unrolling
__global__ void monte_carlo_kernel_optimized(int M, int N, int k, double x, 
                                             unsigned long seed, int *d_S_values, 
                                             int *d_count_reduction) {
    // Shared memory for block-level reduction of count
    __shared__ int s_count[256];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // Initialize shared memory
    s_count[tid] = 0;
    
    if (idx >= M) {
        // Out-of-bounds threads: just participate in reduction with 0
        __syncthreads();
        // Participate in reduction
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                s_count[tid] += s_count[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0) {
            atomicAdd(d_count_reduction, s_count[0]);
        }
        return;
    }

    // Use Philox generator for faster RNG (better performance than default)
    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, 0, &state);

    // Thread-local arrays (registers/local memory)
    const int MAX_N = 64; 
    double Z[MAX_N];
    double R[MAX_N];
    
    // Ensure N is within bounds
    if (N > MAX_N) {
        return; 
    }
    
    // Generate Z (Standard Normal) - optimized: use Philox generator (faster than default)
    // Philox generator has better performance characteristics for parallel workloads
    for (int i = 0; i < N; i++) {
        Z[i] = curand_normal_double(&state);
    }

    // R = mu + L * Z
    // Optimized: Use constant memory, manual unrolling for small N
    // Constant memory is cached and has lower latency than global memory
    // Note: L matrix is stored with stride N (row-major, contiguous)
    if (N == 15) {
        // Manual unrolling for N=15 (common case) - eliminates loop overhead
        // This allows compiler to optimize better and reduces instruction count
        // Using stride N=15 for correct indexing
        double sum;
        
        // Row 0
        sum = c_L[0 * 15 + 0] * Z[0];
        R[0] = c_mu[0] + sum;
        
        // Row 1
        sum = c_L[1 * 15 + 0] * Z[0] + c_L[1 * 15 + 1] * Z[1];
        R[1] = c_mu[1] + sum;
        
        // Row 2
        sum = c_L[2 * 15 + 0] * Z[0] + c_L[2 * 15 + 1] * Z[1] + c_L[2 * 15 + 2] * Z[2];
        R[2] = c_mu[2] + sum;
        
        // Row 3
        sum = c_L[3 * 15 + 0] * Z[0] + c_L[3 * 15 + 1] * Z[1] + c_L[3 * 15 + 2] * Z[2] + c_L[3 * 15 + 3] * Z[3];
        R[3] = c_mu[3] + sum;
        
        // Row 4
        sum = c_L[4 * 15 + 0] * Z[0] + c_L[4 * 15 + 1] * Z[1] + c_L[4 * 15 + 2] * Z[2] + c_L[4 * 15 + 3] * Z[3] + c_L[4 * 15 + 4] * Z[4];
        R[4] = c_mu[4] + sum;
        
        // Row 5
        sum = c_L[5 * 15 + 0] * Z[0] + c_L[5 * 15 + 1] * Z[1] + c_L[5 * 15 + 2] * Z[2] + c_L[5 * 15 + 3] * Z[3] + c_L[5 * 15 + 4] * Z[4] + c_L[5 * 15 + 5] * Z[5];
        R[5] = c_mu[5] + sum;
        
        // Row 6
        sum = c_L[6 * 15 + 0] * Z[0] + c_L[6 * 15 + 1] * Z[1] + c_L[6 * 15 + 2] * Z[2] + c_L[6 * 15 + 3] * Z[3] + c_L[6 * 15 + 4] * Z[4] + c_L[6 * 15 + 5] * Z[5] + c_L[6 * 15 + 6] * Z[6];
        R[6] = c_mu[6] + sum;
        
        // Row 7
        sum = c_L[7 * 15 + 0] * Z[0] + c_L[7 * 15 + 1] * Z[1] + c_L[7 * 15 + 2] * Z[2] + c_L[7 * 15 + 3] * Z[3] + c_L[7 * 15 + 4] * Z[4] + c_L[7 * 15 + 5] * Z[5] + c_L[7 * 15 + 6] * Z[6] + c_L[7 * 15 + 7] * Z[7];
        R[7] = c_mu[7] + sum;
        
        // Row 8
        sum = c_L[8 * 15 + 0] * Z[0] + c_L[8 * 15 + 1] * Z[1] + c_L[8 * 15 + 2] * Z[2] + c_L[8 * 15 + 3] * Z[3] + c_L[8 * 15 + 4] * Z[4] + c_L[8 * 15 + 5] * Z[5] + c_L[8 * 15 + 6] * Z[6] + c_L[8 * 15 + 7] * Z[7] + c_L[8 * 15 + 8] * Z[8];
        R[8] = c_mu[8] + sum;
        
        // Row 9
        sum = c_L[9 * 15 + 0] * Z[0] + c_L[9 * 15 + 1] * Z[1] + c_L[9 * 15 + 2] * Z[2] + c_L[9 * 15 + 3] * Z[3] + c_L[9 * 15 + 4] * Z[4] + c_L[9 * 15 + 5] * Z[5] + c_L[9 * 15 + 6] * Z[6] + c_L[9 * 15 + 7] * Z[7] + c_L[9 * 15 + 8] * Z[8] + c_L[9 * 15 + 9] * Z[9];
        R[9] = c_mu[9] + sum;
        
        // Row 10
        sum = c_L[10 * 15 + 0] * Z[0] + c_L[10 * 15 + 1] * Z[1] + c_L[10 * 15 + 2] * Z[2] + c_L[10 * 15 + 3] * Z[3] + c_L[10 * 15 + 4] * Z[4] + c_L[10 * 15 + 5] * Z[5] + c_L[10 * 15 + 6] * Z[6] + c_L[10 * 15 + 7] * Z[7] + c_L[10 * 15 + 8] * Z[8] + c_L[10 * 15 + 9] * Z[9] + c_L[10 * 15 + 10] * Z[10];
        R[10] = c_mu[10] + sum;
        
        // Row 11
        sum = c_L[11 * 15 + 0] * Z[0] + c_L[11 * 15 + 1] * Z[1] + c_L[11 * 15 + 2] * Z[2] + c_L[11 * 15 + 3] * Z[3] + c_L[11 * 15 + 4] * Z[4] + c_L[11 * 15 + 5] * Z[5] + c_L[11 * 15 + 6] * Z[6] + c_L[11 * 15 + 7] * Z[7] + c_L[11 * 15 + 8] * Z[8] + c_L[11 * 15 + 9] * Z[9] + c_L[11 * 15 + 10] * Z[10] + c_L[11 * 15 + 11] * Z[11];
        R[11] = c_mu[11] + sum;
        
        // Row 12
        sum = c_L[12 * 15 + 0] * Z[0] + c_L[12 * 15 + 1] * Z[1] + c_L[12 * 15 + 2] * Z[2] + c_L[12 * 15 + 3] * Z[3] + c_L[12 * 15 + 4] * Z[4] + c_L[12 * 15 + 5] * Z[5] + c_L[12 * 15 + 6] * Z[6] + c_L[12 * 15 + 7] * Z[7] + c_L[12 * 15 + 8] * Z[8] + c_L[12 * 15 + 9] * Z[9] + c_L[12 * 15 + 10] * Z[10] + c_L[12 * 15 + 11] * Z[11] + c_L[12 * 15 + 12] * Z[12];
        R[12] = c_mu[12] + sum;
        
        // Row 13
        sum = c_L[13 * 15 + 0] * Z[0] + c_L[13 * 15 + 1] * Z[1] + c_L[13 * 15 + 2] * Z[2] + c_L[13 * 15 + 3] * Z[3] + c_L[13 * 15 + 4] * Z[4] + c_L[13 * 15 + 5] * Z[5] + c_L[13 * 15 + 6] * Z[6] + c_L[13 * 15 + 7] * Z[7] + c_L[13 * 15 + 8] * Z[8] + c_L[13 * 15 + 9] * Z[9] + c_L[13 * 15 + 10] * Z[10] + c_L[13 * 15 + 11] * Z[11] + c_L[13 * 15 + 12] * Z[12] + c_L[13 * 15 + 13] * Z[13];
        R[13] = c_mu[13] + sum;
        
        // Row 14
        sum = c_L[14 * 15 + 0] * Z[0] + c_L[14 * 15 + 1] * Z[1] + c_L[14 * 15 + 2] * Z[2] + c_L[14 * 15 + 3] * Z[3] + c_L[14 * 15 + 4] * Z[4] + c_L[14 * 15 + 5] * Z[5] + c_L[14 * 15 + 6] * Z[6] + c_L[14 * 15 + 7] * Z[7] + c_L[14 * 15 + 8] * Z[8] + c_L[14 * 15 + 9] * Z[9] + c_L[14 * 15 + 10] * Z[10] + c_L[14 * 15 + 11] * Z[11] + c_L[14 * 15 + 12] * Z[12] + c_L[14 * 15 + 13] * Z[13] + c_L[14 * 15 + 14] * Z[14];
        R[14] = c_mu[14] + sum;
    } else {
        // Fallback for N != 15: use loop with unroll hint and correct stride
        for (int i = 0; i < N; i++) {
            double sum = 0.0;
            #pragma unroll
            for (int j = 0; j <= i; j++) {
                sum += c_L[i * N + j] * Z[j];
            }
            R[i] = c_mu[i] + sum;
        }
    }

    // Count crashes - optimized: reduce branch divergence using predicated instructions
    // Use bit manipulation to count without branches when possible
    int S = 0;
    const double threshold = -x;
    
    // Unroll crash counting loop for N=15
    if (N == 15) {
        S += (R[0] < threshold) ? 1 : 0;
        S += (R[1] < threshold) ? 1 : 0;
        S += (R[2] < threshold) ? 1 : 0;
        S += (R[3] < threshold) ? 1 : 0;
        S += (R[4] < threshold) ? 1 : 0;
        S += (R[5] < threshold) ? 1 : 0;
        S += (R[6] < threshold) ? 1 : 0;
        S += (R[7] < threshold) ? 1 : 0;
        S += (R[8] < threshold) ? 1 : 0;
        S += (R[9] < threshold) ? 1 : 0;
        S += (R[10] < threshold) ? 1 : 0;
        S += (R[11] < threshold) ? 1 : 0;
        S += (R[12] < threshold) ? 1 : 0;
        S += (R[13] < threshold) ? 1 : 0;
        S += (R[14] < threshold) ? 1 : 0;
    } else {
        // Fallback for N != 15
        for (int i = 0; i < N; i++) {
            S += (R[i] < threshold) ? 1 : 0;
        }
    }

    // Store S value
    d_S_values[idx] = S;
    
    // Use shared memory reduction instead of direct atomic (reduces atomic contention)
    s_count[tid] = (S >= k) ? 1 : 0;
    __syncthreads();
    
    // Block-level reduction using shared memory (tree reduction)
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_count[tid] += s_count[tid + stride];
        }
        __syncthreads();
    }
    
    // Only thread 0 in each block does atomic add (reduces contention from 256x to 1x per block)
    if (tid == 0) {
        atomicAdd(d_count_reduction, s_count[0]);
    }
}

// Host wrapper with manual profiling using CUDA Events
extern "C" int cuda_simulate(MonteCarloParams *params, MonteCarloResult *result) {
    printf("Starting CUDA Monte Carlo simulation (OPTIMIZED) with M = %d trials...\n", params->M);
    
    if (params->N > 64) {
        fprintf(stderr, "Error: N > 64 not supported in this CUDA kernel implementation\n");
        return -1;
    }

    // Create CUDA events for detailed timing
    cudaEvent_t start_total, stop_total;
    cudaEvent_t start_memcpy_h2d, stop_memcpy_h2d;
    cudaEvent_t start_kernel, stop_kernel;
    cudaEvent_t start_memcpy_d2h, stop_memcpy_d2h;
    
    cudaEventCreate(&start_total);
    cudaEventCreate(&stop_total);
    cudaEventCreate(&start_memcpy_h2d);
    cudaEventCreate(&stop_memcpy_h2d);
    cudaEventCreate(&start_kernel);
    cudaEventCreate(&stop_kernel);
    cudaEventCreate(&start_memcpy_d2h);
    cudaEventCreate(&stop_memcpy_d2h);

    // Start total timer
    cudaEventRecord(start_total);

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
    // Note: Using constant memory (c_mu, c_L) for data access, so no need for d_mu/d_L
    int *d_S_values, *d_count;
    
    if (cudaMalloc(&d_S_values, params->M * sizeof(int)) != cudaSuccess) return -1;
    if (cudaMalloc(&d_count, sizeof(int)) != cudaSuccess) return -1;

    // 3. Copy Data (H2D) - Profile memory transfer
    // Copy to constant memory for faster access (cached, low latency)
    cudaEventRecord(start_memcpy_h2d);
    cudaMemcpyToSymbol(c_mu, h_mu_flat, params->N * sizeof(double), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(c_L, h_L_flat, params->N * params->N * sizeof(double), 0, cudaMemcpyHostToDevice);
    cudaMemset(d_count, 0, sizeof(int));
    cudaEventRecord(stop_memcpy_h2d);
    cudaEventSynchronize(stop_memcpy_h2d);

    // 4. Launch Kernel - Profile kernel execution
    int blockSize = 256;
    int numBlocks = (params->M + blockSize - 1) / blockSize;
    
    cudaEventRecord(start_kernel);
    monte_carlo_kernel_optimized<<<numBlocks, blockSize>>>(params->M, params->N, params->k, params->x, 
                                                          params->random_seed, 
                                                          d_S_values, d_count);
    cudaEventRecord(stop_kernel);
    cudaEventSynchronize(stop_kernel);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
         fprintf(stderr, "CUDA Kernel Failed: %s\n", cudaGetErrorString(err));
         return -1;
    }

    // 5. Copy Results Back (D2H) - Profile memory transfer
    result->S_values = (int*)malloc(params->M * sizeof(int));
    cudaEventRecord(start_memcpy_d2h);
    cudaMemcpy(result->S_values, d_S_values, params->M * sizeof(int), cudaMemcpyDeviceToHost);
    
    int h_count;
    cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop_memcpy_d2h);
    cudaEventSynchronize(stop_memcpy_d2h);
    
    result->count = h_count;

    // Stop total timer
    cudaEventRecord(stop_total);
    cudaEventSynchronize(stop_total);

    // Calculate elapsed times
    float time_memcpy_h2d = 0, time_kernel = 0, time_memcpy_d2h = 0, time_total = 0;
    cudaEventElapsedTime(&time_memcpy_h2d, start_memcpy_h2d, stop_memcpy_h2d);
    cudaEventElapsedTime(&time_kernel, start_kernel, stop_kernel);
    cudaEventElapsedTime(&time_memcpy_d2h, start_memcpy_d2h, stop_memcpy_d2h);
    cudaEventElapsedTime(&time_total, start_total, stop_total);

    // Print detailed profiling information
    printf("\n=== CUDA PROFILING BREAKDOWN (OPTIMIZED) ===\n");
    printf("Memory Transfer (H->D):      %.3f ms (%.1f%%)\n", 
           time_memcpy_h2d, (time_memcpy_h2d/time_total)*100);
    printf("  - mu vector:               %zu bytes\n", params->N * sizeof(double));
    printf("  - L matrix:                %zu bytes\n", params->N * params->N * sizeof(double));
    printf("Kernel Execution:            %.3f ms (%.1f%%)\n", 
           time_kernel, (time_kernel/time_total)*100);
    printf("  - Blocks:                  %d\n", numBlocks);
    printf("  - Threads per block:        %d\n", blockSize);
    printf("  - Total threads:           %d\n", numBlocks * blockSize);
    printf("  - Throughput:              %.2f M sim/s\n", params->M / (time_kernel / 1000.0) / 1e6);
    printf("Memory Transfer (D->H):      %.3f ms (%.1f%%)\n", 
           time_memcpy_d2h, (time_memcpy_d2h/time_total)*100);
    printf("  - S_values array:          %zu bytes\n", params->M * sizeof(int));
    printf("  - count value:             %zu bytes\n", sizeof(int));
    printf("Total GPU Time:              %.3f ms\n", time_total);
    printf("============================================\n\n");

    // 6. Stats
    result->P_hat = (double)result->count / params->M;
    result->std_error = sqrt(result->P_hat * (1.0 - result->P_hat) / params->M);
    double margin = 1.96 * result->std_error;
    result->ci_lower = fmax(0.0, result->P_hat - margin);
    result->ci_upper = fmin(1.0, result->P_hat + margin);

    // Cleanup
    cudaFree(d_S_values);
    cudaFree(d_count);
    free(h_mu_flat);
    free(h_L_flat);
    gsl_matrix_free(L);
    
    // Destroy CUDA events
    cudaEventDestroy(start_total);
    cudaEventDestroy(stop_total);
    cudaEventDestroy(start_memcpy_h2d);
    cudaEventDestroy(stop_memcpy_h2d);
    cudaEventDestroy(start_kernel);
    cudaEventDestroy(stop_kernel);
    cudaEventDestroy(start_memcpy_d2h);
    cudaEventDestroy(stop_memcpy_d2h);

    printf("CUDA Simulation complete (OPTIMIZED)\n");
    return 0;
}

extern "C" ModelFunctions get_cuda_model(void) {
    ModelFunctions model = {
        .name = "cuda",
        .run_model = cuda_simulate
    };
    return model;
}

