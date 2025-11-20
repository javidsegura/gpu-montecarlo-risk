// Main entry point for running Monte Carlo simulations with different implementations
// Supports: Serial, OpenMP (future), CUDA (future)

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "model_interface.h"
#include "utilities/load_binary.h"
#include "utilities/load_config.h"
#include "utilities/csv_writer.h"
#include <math.h> //for NaN

extern ModelFunctions get_serial_model(void);
extern ModelFunctions get_openmp_model(void);
extern ModelFunctions get_cuda_model(void);

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

// Get user comment
char* get_user_comment() {
    char *comment = (char *)malloc(256);
    if (!comment) {
        return NULL;
    }

    printf("\nEnter a comment for this simulation (max 255 chars, or press Enter to skip): ");
    fflush(stdout);

    if (fgets(comment, 256, stdin) == NULL) {
        free(comment);
        return NULL;
    }

    // Remove trailing newline
    size_t len = strlen(comment);
    if (len > 0 && comment[len - 1] == '\n') {
        comment[len - 1] = '\0';
    }

    return comment;
}

// Get system information (check how to handle it later)
void get_system_info(int *nodes, int *threads, int *processes) {
    // Placeholder implementation
    *nodes = 0;
    *threads = 0;
    *processes = 0;
}

// Get the next iteration number (thread-safe counter)
int get_next_iteration_number() {
    static int iteration_counter = 0;
    return iteration_counter++;
}

int main() {
    printf("Monte Carlo Financial Risk Simulation\n\n");
    
    printf("Loading parameters from binary files...\n");

    // Load mu and Sigma and actual_freq from binary files
    gsl_vector *mu = load_mu_binary("data/mu.bin");
    gsl_matrix *Sigma = load_sigma_binary("data/sigma.bin");
    double actual_freq = load_actual_freq_binary("data/params.bin");

    if (!mu || !Sigma) {
        fprintf(stderr, "Error: Failed to load binary files. Run Python preprocessing first.\n");
        if (mu) gsl_vector_free(mu);
        if (Sigma) gsl_matrix_free(Sigma);
        return 1;
    }

    if (actual_freq < 0.0) {
        fprintf(stderr, "Warning: Failed to load actual_freq from params.bin, using NaN\n");
        actual_freq = NAN;
    }


    int N = mu->size;  // Number of assets

    // Load configuration parameters from config.yaml
    ConfigParams config;
    if (load_config("config.yaml", &config) != 0) {
        fprintf(stderr, "Error: Failed to load config.yaml. Exiting.\n");
        gsl_vector_free(mu);
        gsl_matrix_free(Sigma);
        return 1;
    }

    int k = config.k;
    double x = config.x;
    int M = config.M;

    printf("\nSimulation Configuration:\n");
    printf("  Number of assets (N): %d\n", N);
    printf("  Crash threshold (k): %d\n", k);
    printf("  Return threshold (x): %.2f%%\n", x * 100);
    printf("  Number of trials (M): %d\n", M);
    printf("  Training ratio: %.2f%%\n", config.train_ratio * 100);

    // Get user comment for this simulation run
    char *user_comment = get_user_comment();

    // Use config comment as fallback if user comment is empty
    char *final_comment = NULL;
    if (user_comment && strlen(user_comment) > 0) {
        final_comment = strdup(user_comment);
    } else if (config.comment && strlen(config.comment) > 0) {
        final_comment = strdup(config.comment);
    } else {
        final_comment = strdup("");
    }
    
    if (user_comment) free(user_comment);

    // Initialize parameters
    MonteCarloParams *params = (MonteCarloParams *)malloc(sizeof(MonteCarloParams));
    if (!params) {
        fprintf(stderr, "Error: Failed to allocate parameters\n");
        gsl_vector_free(mu);
        gsl_matrix_free(Sigma);
        if (final_comment) free(final_comment);
        free_config(&config);
        return 1;
    }

    params->N = N;
    params->k = k;
    params->x = x;
    params->M = M;
    params->random_seed = config.random_seed;
    params->mu = mu;
    params->Sigma = Sigma;

    // Generate unique iteration ID (incremental counter)
    int iteration_id = get_next_iteration_number();

    // Get system information
    int nodes, threads, processes;
    get_system_info(&nodes, &threads, &processes);

    // Models to run - TODO: later read from config
    // const char *models_to_run[] = {"serial", "openmp"};
    // int num_models = sizeof(models_to_run) / sizeof(models_to_run[0]);

    // Run all models from config
    for (int i = 0; i < config.num_models; i++) {
        const char *model_type = config.models[i];
        MonteCarloResult result = {0};
        ModelFunctions model;

        printf("\n=== Running %s model ===\n", model_type);

        // Select model based on type
        if (strcmp(model_type, "serial") == 0) {
            model = get_serial_model();
        }
        else if (strcmp(model_type, "openmp") == 0) {
            model = get_openmp_model();
        }
        else if (strcmp(model_type, "cuda") == 0 || strcmp(model_type, "gpu") == 0) {
            model = get_cuda_model();
        }
        else {
            fprintf(stderr, "Error: Unknown model type '%s'\n", model_type);
            continue;
        }

        // Record start time (wall-clock time for accurate GPU timing)
        struct timespec start_time, end_time;
        clock_gettime(CLOCK_MONOTONIC, &start_time);

        // Run model
        int status = model.run_model(params, &result);

        // Calculate execution time in milliseconds (wall-clock time)
        clock_gettime(CLOCK_MONOTONIC, &end_time);
        long execution_time_ms = (long)((end_time.tv_sec - start_time.tv_sec) * 1000L +
                                        (end_time.tv_nsec - start_time.tv_nsec) / 1000000L);

        if (status == 0) {
            print_results(model.name, &result, M);

            // Write results to CSV
            SimulationResultsData results_data = {
                .iteration_id = iteration_id,
                .timestamp = (long)time(NULL),
                .execution_time_ms = execution_time_ms,
                .comment = final_comment,
                .start_date = config.start,
                .end_date = config.end,
                .train_ratio = config.train_ratio,
                .M = M,
                .k = k,
                .x = x,
                .model_name = model.name,
                .seed = config.random_seed,  // Random seed from config
                .nodes = nodes,
                .threads = threads,
                .processes = processes,
                .indices = config.indices,
                .num_indices = config.num_indices,
                .actual_freq = actual_freq,
                .P_hat = result.P_hat,
                .count = result.count,
                .std_error = result.std_error,
                .ci_lower = result.ci_lower,
                .ci_upper = result.ci_upper
            };

            if (write_results_to_csv("results/simulation_results.csv", &results_data) != 0) {
                fprintf(stderr, "Warning: Failed to write results to CSV\n");
            } else {
                printf("Results written to results/simulation_results.csv\n");
            }
        } else {
            fprintf(stderr, "Error: %s model failed\n", model.name);
        }

        // Always cleanup S_values regardless of success/failure
        make_clean(&result);

        printf("\n");
    }

    free_params(params);
    free_config(&config);
    if (final_comment) free(final_comment);

    printf("Simulation complete.\n");
    return 0;
}
