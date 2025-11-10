// Main entry point for running Monte Carlo simulations with different implementations
// Supports: Serial, OpenMP (future), CUDA (future)
// Reads simulation parameters from config.yaml

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "model_interface.h"
#include "config.h"
#include "csv_io.h"

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

    // Load configuration from YAML
    SimulationConfig config;
    if (load_config("config.yaml", &config) != 0) {
        fprintf(stderr, "Error: Failed to load config.yaml\n");
        return 1;
    }

    printf("Configuration loaded:\n");
    printf("  N=%d, k=%d, x=%.4f, M=%d\n", config.N, config.k, config.x, config.M);
    printf("  mu_file=%s, sigma_file=%s\n", config.mu_file, config.sigma_file);
    printf("  Comment: %s\n\n", config.comment);

    // Initialize parameters
    MonteCarloParams *params = (MonteCarloParams *)malloc(sizeof(MonteCarloParams));
    if (!params) {
        fprintf(stderr, "Error: Failed to allocate parameters\n");
        return 1;
    }

    params->N = config.N;
    params->k = config.k;
    params->x = config.x;
    params->M = config.M;

    // Load mu and Sigma from CSV files
    int ret = read_csv_parameters(config.mu_file, config.sigma_file, config.N,
                                   &params->mu, &params->Sigma);
    if (ret != 0) {
        fprintf(stderr, "Error: Failed to load CSV files (error code: %d)\n", ret);
        free(params);
        return 1;
    }

    printf("Parameters loaded from CSV files successfully\n\n");

    // Models to run - TODO: later read from config
    const char *models_to_run[] = {"serial", "openmp"};
    
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
