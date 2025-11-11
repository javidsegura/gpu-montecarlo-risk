// Monte Carlo simulation for financial crash probability - OPENMP VERSION
//
// OPTIMIZATION STRATEGY:
// 1. Thread-local workspace: Allocate Z and R vectors ONCE per thread (not per iteration)
// 2. Static scheduling: Better cache locality and lower overhead than dynamic: Monte Carlo trials have uniform cost, no load balancing needed
// 3. Per-thread RNGs
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
    gsl_rng **rng_array;    // Array of RNGs, one per thread
    int num_threads;
} OpenMPModelState;

// Initialize OpenMP model: Cholesky decomposition + per-thread RNG setup
static int openmp_init(MonteCarloParams *params, void **model_state) {
    OpenMPModelState *state = (OpenMPModelState *)malloc(sizeof(OpenMPModelState));
    if (!state) {
        fprintf(stderr, "Error: Failed to allocate OpenMPModelState\n");
        return -1;
    }

    state->num_threads = omp_get_max_threads();
    printf("Using %d OpenMP threads\n", state->num_threads);

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

    // Initialize one RNG per thread 
    state->rng_array = (gsl_rng **)malloc(state->num_threads * sizeof(gsl_rng *));
    if (!state->rng_array) {
        fprintf(stderr, "Error: Failed to allocate RNG array\n");
        gsl_matrix_free(state->L);
        free(state);
        return -1;
    }

    for (int i = 0; i < state->num_threads; i++) {
        state->rng_array[i] = gsl_rng_alloc(gsl_rng_mt19937);
        gsl_rng_set(state->rng_array[i], RNG_SEED);  // Same seed for all threads
    }

    *model_state = state;
    return 0;
}

// Run a single trial - thread-safe version using pre-allocated thread-local vectors
// Called from within parallel loop with thread-local workspace
static int openmp_run_trial(void *model_state, MonteCarloParams *params, 
                            gsl_vector *Z, gsl_vector *R, int *S_out) {
    OpenMPModelState *state = (OpenMPModelState *)model_state;
    int thread_id = omp_get_thread_num();

    // Generate N independent standard normal variates (using thread's RNG)
    for (int i = 0; i < params->N; i++) {
        gsl_vector_set(Z, i, gsl_ran_gaussian(state->rng_array[thread_id], 1.0));
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

// Cleanup internal model state (RNGs, Cholesky)
static void openmp_cleanup_state(void *model_state) {
    if (model_state) {
        OpenMPModelState *state = (OpenMPModelState *)model_state;
        if (state->rng_array) {
            for (int i = 0; i < state->num_threads; i++) {
                if (state->rng_array[i]) gsl_rng_free(state->rng_array[i]);
            }
            free(state->rng_array);
        }
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

    // STEP 2: Initialize model resources (Cholesky + per-thread RNGs)
    void *model_state = NULL;
    int status = openmp_init(params, &model_state);
    if (status != 0) {
        free(result->S_values);
        return -1;
    }

    // STEP 3: Run M trials in PARALLEL using OpenMP
    int count = 0;

    #pragma omp parallel
    {
        // Allocate thread-local workspace ONCE per thread (major optimization)
        gsl_vector *Z = gsl_vector_alloc(params->N);
        gsl_vector *R = gsl_vector_alloc(params->N);

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

        // Free thread-local workspace
        gsl_vector_free(Z);
        gsl_vector_free(R);
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
