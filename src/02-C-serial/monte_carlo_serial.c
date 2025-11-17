// Monte Carlo simulation for financial crash probability - SERIAL VERSION
// Replicates Python serial implementation using GSL

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include "../model_interface.h"

typedef struct {
    gsl_matrix *L;      // Cholesky decomposition of covariance matrix
    gsl_rng *rng;       // Random number generator
} SerialModelState;

// Initialize serial model: Cholesky decomposition + RNG setup
static int serial_init(MonteCarloParams *params, void **model_state) {
    SerialModelState *state = (SerialModelState *)malloc(sizeof(SerialModelState));
    if (!state) {
        fprintf(stderr, "Error: Failed to allocate SerialModelState\n");
        return -1;
    }

    // Precompute Cholesky decomposition L such that Sigma = L * L^T
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

    // Initialize random number generator
        state->rng = gsl_rng_alloc(gsl_rng_mt19937);
    if (!state->rng) {
        fprintf(stderr, "Error: Failed to allocate RNG\n");
        gsl_matrix_free(state->L);
        free(state);
        return -1;
    }
    gsl_rng_set(state->rng, params->random_seed);

    *model_state = state;
    return 0;
}

// Run a single trial - generate correlated returns and count crashes
// Z and R are pre-allocated workspace vectors to avoid repeated allocation overhead
static int serial_run_trial(void *model_state, MonteCarloParams *params,
                            gsl_vector *Z, gsl_vector *R, int *S_out) {
    SerialModelState *state = (SerialModelState *)model_state;

    // Generate N independent standard normal variates
    for (int i = 0; i < params->N; i++) {
        gsl_vector_set(Z, i, gsl_ran_gaussian(state->rng, 1.0));
    }

    // Transform to correlated returns: R = mu + L*Z
    gsl_blas_dgemv(CblasNoTrans, 1.0, state->L, Z, 0.0, R);
    gsl_vector_add(R, params->mu);

    // Count how many assets crashed (return falls below threshold -x)
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
static void serial_cleanup_state(void *model_state) {
    if (model_state) {
        SerialModelState *state = (SerialModelState *)model_state;
        if (state->rng) gsl_rng_free(state->rng);
        if (state->L) gsl_matrix_free(state->L);
        free(state);
    }
}

// Main simulation
static int serial_simulate(MonteCarloParams *params, MonteCarloResult *result) {
    printf("Starting SERIAL Monte Carlo simulation with M = %d trials...\n", params->M);
    printf("Parameters: N=%d, k=%d, x=%.2f%%\n", params->N, params->k, params->x * 100);

    // STEP 1: Allocate result arrays
    result->count = 0;
    result->S_values = (int *)calloc(params->M, sizeof(int));
    if (!result->S_values) {
        fprintf(stderr, "Error: Failed to allocate S_values array\n");
        return -1;
    }

    // STEP 2: Initialize model resources (Cholesky decomposition + RNG)
    void *model_state = NULL;
    int status = serial_init(params, &model_state);
    if (status != 0) {
        free(result->S_values);
        return -1;
    }

    // STEP 3: Allocate workspace vectors once to avoid repeated allocation overhead
    gsl_vector *Z = gsl_vector_alloc(params->N);  // Standard normal random variables
    gsl_vector *R = gsl_vector_alloc(params->N);  // Correlated returns
    if (!Z || !R) {
        fprintf(stderr, "Error: Failed to allocate workspace vectors\n");
        if (Z) gsl_vector_free(Z);
        if (R) gsl_vector_free(R);
        serial_cleanup_state(model_state);
        free(result->S_values);
        return -1;
    }

    // STEP 4: Run M independent trials
    for (int j = 0; j < params->M; j++) {
        int S = 0;

        // Run one trial using pre-allocated workspace
        serial_run_trial(model_state, params, Z, R, &S);

        result->S_values[j] = S;

        // Check if this trial resulted in an extreme event (S >= k)
        if (S >= params->k) {
            result->count++;
        }
    }

    // Free workspace vectors
    gsl_vector_free(Z);
    gsl_vector_free(R);

    printf("Simulation complete\n");

    // STEP 5: Calculate final probability estimate
    result->P_hat = (double)result->count / params->M;

    // STEP 6: Calculate accuracy metrics (standard error and 95% CI)
    result->std_error = sqrt(result->P_hat * (1.0 - result->P_hat) / params->M);
    double margin = 1.96 * result->std_error;
    result->ci_lower = fmax(0.0, result->P_hat - margin);
    result->ci_upper = fmin(1.0, result->P_hat + margin);

    // STEP 7: Cleanup internal state (result cleanup handled by main_runner)
    serial_cleanup_state(model_state);

    return 0;
}

// Return ModelFunctions struct for this model
ModelFunctions get_serial_model(void) {
    ModelFunctions model = {
        .name = "serial",
        .run_model = serial_simulate
    };
    return model;
}
