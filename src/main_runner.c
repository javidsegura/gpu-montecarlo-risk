// Main entry point for running Monte Carlo simulations with different implementations
// Supports: Serial, OpenMP (future), CUDA (future)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "model_interface.h"

extern ModelFunctions get_serial_model(void);
extern ModelFunctions get_openmp_model(void);
// extern ModelFunctions get_cuda_model(void);

// Print final results
void print_results(const char *model_name, MonteCarloResult *result, int M) {
    printf("\n%s RESULTS\n", model_name);
    printf("Estimated Probability (P_MIN): %.6f\n", result->P_hat);
    printf("Extreme Events Count: %d out of %d trials\n", result->count, M);
    printf("Standard Error: %.6f\n", result->std_error);
    printf("95%% Confidence Interval: [%.6f, %.6f]\n", result->ci_lower, result->ci_upper);
}

// Free parameters
void free_params(MonteCarloParams *params) {
    if (params) {
        if (params->mu) gsl_vector_free(params->mu);
        if (params->Sigma) gsl_matrix_free(params->Sigma);
        free(params);
    }
}

// Cleanup result arrays - standard cleanup for all models
void make_clean(MonteCarloResult *result) {
    if (result && result->S_values) {
        free(result->S_values);
        result->S_values = NULL;
    }
}

int main() {
    printf("Monte Carlo Financial Risk Simulation\n\n");

    // -----------------------------------------------------------------------
    // ALL PARAMATERS AND HARDCODED VALS -> TODO read data and params
    // Simulation parameters (matching Python)
    int N = 10;           // Number of assets
    int k = 5;            // Crash threshold
    double x = 0.02;      // Return threshold (2%)
    int M = 100000;       // Number of trials
    double rho = 0.3;     // Correlation coefficient
    double variance = 0.04;

    // Initialize parameters
    MonteCarloParams *params = (MonteCarloParams *)malloc(sizeof(MonteCarloParams));
    if (!params) {
        fprintf(stderr, "Error: Failed to allocate parameters\n");
        return 1;
    }

    params->N = N;
    params->k = k;
    params->x = x;
    params->M = M;
    params->mu = gsl_vector_calloc(N);

    params->Sigma = gsl_matrix_alloc(N, N);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i == j) {
                gsl_matrix_set(params->Sigma, i, j, variance);
            } else {
                gsl_matrix_set(params->Sigma, i, j, rho * variance);
            }
        }
    }

    // Models to run - TODO: later read from config
    const char *models_to_run[] = {"serial", "openmp"};
    // --------------------------------------------------------------------
    
    int num_models = sizeof(models_to_run) / sizeof(models_to_run[0]);

    // Run all models
    for (int i = 0; i < num_models; i++) {
        const char *model_type = models_to_run[i];
        MonteCarloResult result = {0};
        ModelFunctions model;

        // Select model based on type
        if (strcmp(model_type, "serial") == 0) {
            model = get_serial_model();
        }
        else if (strcmp(model_type, "openmp") == 0) {
            model = get_openmp_model();
        }
        else {
            fprintf(stderr, "Error: Unknown model type '%s'\n", model_type);
            continue;
        }

        // Run model
        int status = model.run_model(params, &result);

        if (status == 0) {
            print_results(model.name, &result, M);
        } else {
            fprintf(stderr, "Error: %s model failed\n", model.name);
        }

        // TODO save results

        // Always cleanup S_values regardless of success/failure
        make_clean(&result);

        printf("\n");
    }

    free_params(params);
    return 0;
}
