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

    // SAACT: Log simulation start with parameters
    fprintf(stderr, "[openmp_simulate] START_SIMULATION | actor=openmp | ctx={N=%d, k=%d, x=%.4f, M=%d, seed=%d} | level=INFO\n", params->N, params->k, params->x, params->M, params->random_seed);

    // STEP 1: Allocate result arrays
    result->count = 0;
    result->S_values = (int *)calloc(params->M, sizeof(int));
    if (!result->S_values) {
        fprintf(stderr, "[openmp_simulate] ALLOCATE_MEMORY | actor=openmp | ctx={M=%d, size_bytes=%zu} | level=ERROR: allocation_failed\n", params->M, params->M * sizeof(int));
        return -1;
    }

    // STEP 2: Initialize model resources (Cholesky + shared RNG)
    void *model_state = NULL;
    int status = openmp_init(params, &model_state);
    if (status != 0) {
        fprintf(stderr, "[openmp_init] INITIALIZE_MODEL | actor=openmp | ctx={N=%d} | level=ERROR: init_failed\n", params->N);
        free(result->S_values);
        return -1;
    }

    // STEP 3: Run M trials in PARALLEL using OpenMP
    int count = 0;
    int allocation_error = 0;

    #pragma omp parallel
    {
        gsl_vector *Z = gsl_vector_alloc(params->N);
        gsl_vector *R = gsl_vector_alloc(params->N);

        if (!Z || !R) {
            #pragma omp atomic write
            allocation_error = 1;
        }

        #pragma omp barrier

        if (!allocation_error) {
            #pragma omp for reduction(+:count) schedule(static)
            for (int j = 0; j < params->M; j++) {
                // Run trial using thread local workspace
                int S = 0;
                openmp_run_trial(model_state, params, Z, R, &S);

                result->S_values[j] = S;

                if (S >= params->k) {
                    count++;
                }
            }
        }


        if (R) gsl_vector_free(R);
        if (Z) gsl_vector_free(Z);
    
    }

    // Check for allocation errors after parallel region
    if (allocation_error) {
        fprintf(stderr, "[openmp_simulate] THREAD_ALLOCATION | actor=openmp | ctx={N=%d, M=%d} | level=ERROR: thread_local_allocation_failed\n", params->N, params->M);
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

    // SAACT: Log simulation end with results
    fprintf(stderr, "[openmp_simulate] END_SIMULATION | actor=openmp | ctx={M=%d, count=%d, P_hat=%.6f, ci_lower=%.6f, ci_upper=%.6f} | level=INFO\n", params->M, result->count, result->P_hat, result->ci_lower, result->ci_upper);

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
