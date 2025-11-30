// Monte Carlo simulation for financial crash probability - MPI+OPENMP HYBRID VERSION
//
// OPTIMIZATION STRATEGY:
// 1. MPI: Distribute M trials across multiple processes (distributed memory)
// 2. OpenMP: Each MPI process uses multiple threads (shared memory)
// 3. Per-thread RNG: Deterministic seeding based on rank + thread_id (NO CRITICAL SECTION)
// 4. Per-thread workspace: Allocate Z and R vectors ONCE per thread (not per iteration)
// 5. MPI_Ireduce: Aggregate results from all processes to rank 0

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include "../model_interface.h"

// MPI+OpenMP hybrid model state
typedef struct {
    gsl_matrix *L;          // Cholesky decomposition (shared by all threads)
    gsl_rng **rng_array;    // Array of per-thread RNGs (no critical section needed)
    int num_threads;

    int rank;               // MPI process rank
    int size;               // Total number of MPI processes
} MPIOpenMPModelState;

// Initialize MPI+OpenMP model: Cholesky decomposition + per-thread RNG setup
static int mpi_openmp_init(MonteCarloParams *params, void **model_state, int rank, int size) {
    MPIOpenMPModelState *state = (MPIOpenMPModelState *)malloc(sizeof(MPIOpenMPModelState));
    if (!state) {
        fprintf(stderr, "[Rank %d] Error: Failed to allocate MPIOpenMPModelState\n", rank);
        return -1;
    }

    state->rank = rank;
    state->size = size;
    state->num_threads = omp_get_max_threads();
    omp_set_num_threads(state->num_threads);
    
    if (rank == 0) {
        printf("MPI+OpenMP Hybrid Configuration:\n");
        printf("    MPI processes: %d\n", size);
        printf("    OpenMP threads per process: %d\n", state->num_threads);
        printf("    Total parallel execution units (workers): %d\n", size * state->num_threads);
    }

    // Precompute Cholesky decomposition (shared by all threads in this process)
    state->L = gsl_matrix_alloc(params->N, params->N);
    if (!state->L) {
        fprintf(stderr, "[Rank %d] Error: Failed to allocate Cholesky matrix\n", rank);
        free(state);
        return -1;
    }

    gsl_matrix_memcpy(state->L, params->Sigma);
    int status = gsl_linalg_cholesky_decomp1(state->L);
    if (status) {
        fprintf(stderr, "[Rank %d] Error: Cholesky decomposition failed\n", rank);
        gsl_matrix_free(state->L);
        free(state);
        return -1;
    }

    // Allocate array of per-thread RNGs
    state->rng_array = (gsl_rng **)malloc(state->num_threads * sizeof(gsl_rng *));
    if (!state->rng_array) {
        fprintf(stderr, "[Rank %d] Error: Failed to allocate RNG array\n", rank);
        gsl_matrix_free(state->L);
        free(state);
        return -1;
    }

    // Initialize one RNG per thread with deterministic seeding
    // Seed = base_seed + rank*1000 + thread_id to ensure different streams across MPI+OpenMP
    for (int i = 0; i < state->num_threads; i++) {
        state->rng_array[i] = gsl_rng_alloc(gsl_rng_mt19937);
        if (!state->rng_array[i]) {
            fprintf(stderr, "[Rank %d] Error: Failed to allocate RNG for thread %d\n", rank, i);
            // Cleanup previously allocated RNGs
            for (int j = 0; j < i; j++) {
                gsl_rng_free(state->rng_array[j]);
            }
            free(state->rng_array);
            gsl_matrix_free(state->L);
            free(state);
            return -1;
        }
        // Deterministic seed: base_seed + rank*1000 + thread_id
        // This ensures each (rank, thread) pair has a unique RNG stream
        gsl_rng_set(state->rng_array[i], params->random_seed + rank * 1000 + i);
    }

    *model_state = state;
    return 0;
}

// Run a single trial - thread-safe version using per-thread RNG (NO critical section)
// Called from within parallel loop with thread-local workspace
static int mpi_openmp_run_trial(void *model_state, MonteCarloParams *params,
                                 gsl_rng *thread_rng, gsl_vector *Z, gsl_vector *R, int *S_out) {
    MPIOpenMPModelState *state = (MPIOpenMPModelState *)model_state;

    // Generate N independent standard normal variates (using per-thread RNG - NO CRITICAL SECTION)
    for (int i = 0; i < params->N; i++) {
        gsl_vector_set(Z, i, gsl_ran_gaussian(thread_rng, 1.0));
    }

    // Transform to correlated returns: R = mu + L*Z
    gsl_blas_dgemv(CblasNoTrans, 1.0, state->L, Z, 0.0, R);
    gsl_vector_add(R, params->mu);

    // Count crashes
    int S = 0;
    for (int i = 0; i < params->N; i++) {
        if (gsl_vector_get(R, i) < -params->x) {
            S++;
        }
    }

    *S_out = S;
    return 0;
}

// Cleanup internal model state (RNG array, Cholesky)
static void mpi_openmp_cleanup_state(void *model_state) {
    if (model_state) {
        MPIOpenMPModelState *state = (MPIOpenMPModelState *)model_state;

        // Free all per-thread RNGs
        if (state->rng_array) {
            for (int i = 0; i < state->num_threads; i++) {
                if (state->rng_array[i]) {
                    gsl_rng_free(state->rng_array[i]);
                }
            }
            free(state->rng_array);
        }
        // Free Cholesky matrix
        if (state->L) {gsl_matrix_free(state->L);}
        free(state);
    }
}

// Main MPI+OpenMP simulation
static int mpi_openmp_simulate(MonteCarloParams *params, MonteCarloResult *result) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        printf("Starting MPI+OpenMP Monte Carlo simulation with M = %d trials...\n", params->M);
        printf("Parameters: N=%d, k=%d, x=%.2f%%\n", params->N, params->k, params->x * 100);
    }

    // STEP 1: Divide work among MPI processes
    int trials_per_rank = params->M / size;
    int remainder = params->M % size; // If M not divisible by size
    
    // How many trials for this rank
    int local_M = trials_per_rank + (rank < remainder ? 1 : 0);

    if (rank == 0) {
        printf("Work distribution:\n");
        for (int r = 0; r < size; r++) { // for each rank 
            int r_M = trials_per_rank + (r < remainder ? 1 : 0); // trials for rank r
            int r_start = r * trials_per_rank + (r < remainder ? r : remainder); // start index for rank r
            printf("  Rank %d: %d trials [%d to %d)\n", r, r_M, r_start, r_start + r_M);
        }
    }

    // Local allocations + init, with a GLOBAL error check
    int *local_S_values = NULL;
    void *model_state = NULL;
    int local_err_init = 0;
    int global_err_init = 0;

    // STEP 2: Allocate local result arrays (only for this rank's trials)
    if (local_M > 0) {
        local_S_values = (int *)calloc(local_M, sizeof(int));
        if (!local_S_values) {
            fprintf(stderr, "[Rank %d] Error: Failed to allocate local S_values array\n", rank);
            local_err_init = 1;
        }
    } else {
        // No trials on this rank; it's valid to have no local_S_values
        local_S_values = NULL;
    }

    // STEP 3: Initialize model resources (Cholesky + per-thread RNGs)
    if (!local_err_init) {
        int status = mpi_openmp_init(params, &model_state, rank, size);
        if (status != 0) {
            // mpi_openmp_init already printed an error
            local_err_init = 1;
        }
    }

    // Make sure ALL ranks know if ANY rank failed before doing compute/collectives
    MPI_Allreduce(&local_err_init, &global_err_init, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    if (global_err_init) {
        // Clean up whatever was allocated on this rank
        if (local_S_values) {
            free(local_S_values);
        }
        if (model_state) {
            mpi_openmp_cleanup_state(model_state);
        }
        // All ranks return the same error code
        return -1;
    }

    // STEP 4: Run local_M trials in PARALLEL using OpenMP
    int local_count = 0;
    MPIOpenMPModelState *state = (MPIOpenMPModelState *)model_state;

    // Start timing the kernel (parallel computation)
    double kernel_start_time = MPI_Wtime();

    // Shared error flag for OpenMP threads
    int thread_err = 0;

    #pragma omp parallel shared(thread_err)
    {
        // Get thread-specific RNG (pre-allocated in init, no malloc here!)
        int thread_id = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        
        if (nthreads != state->num_threads) {
            // This should never happen since we set num_threads in init
            #pragma omp single
            {
                fprintf(stderr, "[Rank %d] Warning: Number of threads changed from %d to %d\n",
                        rank, state->num_threads, nthreads);
            }
            // Mark an error (atomic to avoid a data race on thread_err)
            #pragma omp atomic write
            thread_err = 1;
        }

        // Allocate thread-local workspace ONCE per thread (major optimization)
        gsl_vector *Z = gsl_vector_alloc(params->N);
        gsl_vector *R = gsl_vector_alloc(params->N);

        #pragma omp barrier 

        if (!thread_err){ 
            gsl_rng *my_rng = state->rng_array[thread_id];
            // Parallel loop over local trials with reduction for local_count
            // Use static scheduling for better cache locality and less overhead
            #pragma omp for reduction(+:local_count) schedule(static)
            for (int j = 0; j < local_M; j++) {
                // Run trial using thread-local RNG and workspace (NO CRITICAL SECTION)
                int S = 0;
                mpi_openmp_run_trial(model_state, params, my_rng, Z, R, &S);
                local_S_values[j] = S;
                // Check if this trial resulted in an extreme event
                if (S >= params->k) {
                    local_count++;
                }
            }
        }

        // Free thread-local workspace (RNGs freed in cleanup)
        gsl_vector_free(Z);
        gsl_vector_free(R);
    }

    // End timing the kernel
    double kernel_end_time = MPI_Wtime();
    double local_kernel_time_ms = (kernel_end_time - kernel_start_time) * 1000.0;
    
    
    printf("[Rank %d] Local computation complete (%.3f ms)\n", rank, local_kernel_time_ms);
    
    // Check if any OpenMP thread on any rank saw a threading error
    int local_err_threads = thread_err;
    int global_err_threads = 0;
    MPI_Allreduce(&local_err_threads, &global_err_threads, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    if (global_err_threads) {
        // Clean up and report backend failure to the caller
        free(local_S_values);
        mpi_openmp_cleanup_state(model_state);
        return -1;
    }

    // STEP 5: Prepare gather buffers and check for allocation errors collectively
    int *recvcounts = NULL;
    int *displs = NULL;
    int *global_S_values = NULL;
    int local_err_gather = 0;

    if (rank == 0) {
        recvcounts = (int *)malloc(size * sizeof(int));
        displs     = (int *)malloc(size * sizeof(int));

        if (!recvcounts || !displs) {
            fprintf(stderr, "[Rank 0] Error: Failed to allocate recvcounts/displs\n");
            local_err_gather = 1;
        } else {
            for (int r = 0; r < size; r++) {
                recvcounts[r] = trials_per_rank + (r < remainder ? 1 : 0);
                displs[r]     = r * trials_per_rank + (r < remainder ? r : remainder);
            }

            global_S_values = (int *)malloc(params->M * sizeof(int));
            if (!global_S_values) {
                fprintf(stderr, "[Rank 0] Error: Failed to allocate global_S_values array\n");
                local_err_gather = 1;
            }
        }
    }

    // Check if any rank encountered an error (so far it's only rank 0 that can set local_err)
    int global_err_gather = 0;
    MPI_Allreduce(&local_err_gather, &global_err_gather, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    if (global_err_gather) {
        // Clean up and return error BEFORE posting any nonblocking collectives
        if (rank == 0) {
            free(recvcounts);
            free(displs);
            free(global_S_values);
        }
        free(local_S_values);
        mpi_openmp_cleanup_state(model_state);
        return -1;
    }

    // Aggregate results across all MPI processes using non-blocking MPI operations
    int global_count = 0;
    MPI_Request reduce_request, gatherv_request, time_request;

    // 1) Non-blocking reduce for global count
    MPI_Ireduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM,
                0, MPI_COMM_WORLD, &reduce_request);

    // 2) Non-blocking gatherv for all S_values to rank 0
    MPI_Igatherv(local_S_values, local_M, MPI_INT,
                 global_S_values, recvcounts, displs, MPI_INT,
                 0, MPI_COMM_WORLD, &gatherv_request);

    // 3) Non-blocking reduce for maximum kernel time across ranks
    double max_kernel_time_ms = 0.0;
    MPI_Ireduce(&local_kernel_time_ms, &max_kernel_time_ms, 1, MPI_DOUBLE, MPI_MAX,
                0, MPI_COMM_WORLD, &time_request);

    // Wait for all MPI requests to complete before processing results
    MPI_Request requests[3] = { reduce_request, gatherv_request, time_request };
    MPI_Waitall(3, requests, MPI_STATUSES_IGNORE);

    // STEP 6: Rank 0 computes final statistics
    if (rank == 0) {
        // Store gathered per-trial S values in the result (caller owns freeing them)
        result->S_values = global_S_values;
        result->count = global_count;
        result->kernel_time_ms = max_kernel_time_ms;
        
        printf("Simulation complete\n");
        printf("    Max kernel time across ranks: %.3f ms\n", result->kernel_time_ms);
        printf("    Total extreme events: %d / %d\n", result->count, params->M);

        // Calculate final probability estimate
        result->P_hat = (double)result->count / params->M;

        // Calculate accuracy metrics (standard error and 95% CI)
        result->std_error = sqrt(result->P_hat * (1.0 - result->P_hat) / params->M);
        double margin = 1.96 * result->std_error;
        result->ci_lower = fmax(0.0, result->P_hat - margin);
        result->ci_upper = fmin(1.0, result->P_hat + margin);

        free(recvcounts);
        free(displs);
    }

    // STEP 7: Cleanup local resources
    free(local_S_values);
    mpi_openmp_cleanup_state(model_state);

    return 0;

}

// Return ModelFunctions struct for this model
ModelFunctions get_mpi_openmp_model(void) {
    ModelFunctions model = {
        .name = "mpi_openmp",
        .run_model = mpi_openmp_simulate
    };
    return model;
}
