// Monte Carlo simulation for financial crash probability - OPENMP VERSION
//
// OPTIMIZATION STRATEGY:
// 1. Thread-local workspace: Allocate Z and R vectors ONCE per thread (not per iteration)
// 2. Static scheduling: Better cache locality and lower overhead than dynamic: Monte Carlo trials have uniform cost, no load balancing needed
// 3. Single shared RNG with critical section
// 4. Reduction clause

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include "../model_interface.h"

// OpenMP model state
typedef struct {
    gsl_matrix *L;          // Cholesky decomposition (shared by all threads)
    gsl_rng *rng;           // Single shared RNG (protected by critical section)
    int num_threads;
} OpenMPModelState;

// Initialize OpenMP model: Cholesky decomposition + shared RNG setup
static int openmp_init(MonteCarloParams *params, void **model_state) {
    OpenMPModelState *state = (OpenMPModelState *)malloc(sizeof(OpenMPModelState));
    if (!state) {
        fprintf(stderr, "Error: Failed to allocate OpenMPModelState\n");
        return -1;
    }

    state->num_threads = omp_get_max_threads();
    printf("Using %d OpenMP threads with shared RNG\n", state->num_threads);

    // Precompute Cholesky decomposition (shared by all threads)
    state->L = gsl_matrix_alloc(params->N, params->N);
    if (!state->L) {
        fprintf(stderr, "Error: Failed to allocate Cholesky matrix\n");
        free(state);
        return -1;
    }

    gsl_matrix_memcpy(state->L, params->Sigma);
    int status = gsl_linalg_cholesky_decomp1(state->L);
    if (status) {
        fprintf(stderr, "Error: Cholesky decomposition failed\n");
        gsl_matrix_free(state->L);
        free(state);
        return -1;
    }

    // Initialize single shared RNG
    state->rng = gsl_rng_alloc(gsl_rng_mt19937);
    if (!state->rng) {
        fprintf(stderr, "Error: Failed to allocate RNG\n");
        gsl_matrix_free(state->L);
        free(state);
        return -1;
    }
    gsl_rng_set(state->rng, params->random_seed);  // Configurable seed

    *model_state = state;
    return 0;
}

// Run a single trial - thread-safe version using pre-allocated thread-local vectors
// Called from within parallel loop with thread-local workspace
static int openmp_run_trial(void *model_state, MonteCarloParams *params,
                            gsl_vector *Z, gsl_vector *R, int *S_out) {
    OpenMPModelState *state = (OpenMPModelState *)model_state;

    // Generate N independent standard normal variates (using shared RNG with critical section)
    #pragma omp critical
    {
        for (int i = 0; i < params->N; i++) {
            gsl_vector_set(Z, i, gsl_ran_gaussian(state->rng, 1.0));
        }
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

// Cleanup internal model state (RNG, Cholesky)
static void openmp_cleanup_state(void *model_state) {
    if (model_state) {
        OpenMPModelState *state = (OpenMPModelState *)model_state;
        if (state->rng) gsl_rng_free(state->rng);
        if (state->L) gsl_matrix_free(state->L);
        free(state);
    }
}

// Main simulation
static int openmp_simulate(MonteCarloParams *params, MonteCarloResult *result) {
    printf("Starting OPENMP Monte Carlo simulation with M = %d trials...\n", params->M);
    printf("Parameters: N=%d, k=%d, x=%.2f%%\n", params->N, params->k, params->x * 100);

    // STEP 1: Allocate result arrays
    result->count = 0;
    result->S_values = (int *)calloc(params->M, sizeof(int));
    if (!result->S_values) {
        fprintf(stderr, "Error: Failed to allocate S_values array\n");
        return -1;
    }

    // STEP 2: Initialize model resources (Cholesky + shared RNG)
    void *model_state = NULL;
    int status = openmp_init(params, &model_state);
    if (status != 0) {
        free(result->S_values);
        return -1;
    }

    // STEP 3: Run M trials in PARALLEL using OpenMP
    int count = 0;
    int allocation_error = 0;

    #pragma omp parallel
    {
        // Allocate thread-local workspace ONCE per thread (major optimization)
        gsl_vector *Z = gsl_vector_alloc(params->N);
        gsl_vector *R = gsl_vector_alloc(params->N);

        // Check for allocation failure
        if (!Z || !R) {
            fprintf(stderr, "Error: Failed to allocate workspace vectors in thread %d\n", omp_get_thread_num());
            if (Z) gsl_vector_free(Z);
            if (R) gsl_vector_free(R);

            // Set error flag and cancel the parallel region
            #pragma omp atomic write
            allocation_error = 1;

            #pragma omp cancel parallel
        }

        // Check if cancellation was requested by another thread
        #pragma omp cancellation point parallel

        // Parallel loop over trials with reduction for count
        // Use static scheduling for better cache locality and less overhead
        #pragma omp for reduction(+:count) schedule(static)
        for (int j = 0; j < params->M; j++) {
            // Run trial using thread-local workspace
            int S = 0;
            openmp_run_trial(model_state, params, Z, R, &S);

            result->S_values[j] = S;

            // Check if this trial resulted in an extreme event
            if (S >= params->k) {
                count++;
            }
        }

        // Free thread-local workspace (safe: only reached if allocation succeeded)
        gsl_vector_free(Z);
        gsl_vector_free(R);
    }

    // Check for allocation errors after parallel region
    if (allocation_error) {
        openmp_cleanup_state(model_state);
        free(result->S_values);
        return -1;
    }

    result->count = count;
    printf("Simulation complete\n");

    // STEP 4: Calculate final probability estimate
    result->P_hat = (double)result->count / params->M;

    // STEP 5: Calculate accuracy metrics (standard error and 95% CI)
    result->std_error = sqrt(result->P_hat * (1.0 - result->P_hat) / params->M);
    double margin = 1.96 * result->std_error;
    result->ci_lower = fmax(0.0, result->P_hat - margin);
    result->ci_upper = fmin(1.0, result->P_hat + margin);

    // STEP 6: Cleanup internal state (result cleanup handled by main_runner)
    openmp_cleanup_state(model_state);

    return 0;
}

// Return ModelFunctions struct for this model
ModelFunctions get_openmp_model(void) {
    ModelFunctions model = {
        .name = "openmp",
        .run_model = openmp_simulate
    };
    return model;
}
