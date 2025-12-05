// Main entry point for running Monte Carlo simulations with different implementations
// Supports: Serial, OpenMP, MPI+OpenMP, CUDA (future)

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <mpi.h>
#include "model_interface.h"
#include "utilities/load_binary.h"
#include "utilities/load_config.h"
#include "utilities/csv_writer.h"
#include <math.h> //for NaN

#ifdef SERIAL_BUILD
extern ModelFunctions get_serial_model(void);
#endif

#ifdef OPENMP_BUILD
extern ModelFunctions get_openmp_model(void);
#endif

#ifdef OPENMP_OPT_BUILD
extern ModelFunctions get_openmp_opt_model(void);
#endif

#ifdef MPI_OPENMP_BUILD
extern ModelFunctions get_mpi_openmp_model(void);
#endif

#ifdef CUDA_BUILD
extern ModelFunctions get_cuda_model(void);
#endif

// Print final results
void print_results(const char *model_name, MonteCarloResult *result, int M) {
    printf("\n%s RESULTS\n", model_name);
    printf("Estimated Probability (P_MIN): %.6f\n", result->P_hat);
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


// Get Slurm system information from environment variables - requested resources
void get_slurm_info(int *slurm_nodes, int *slurm_threads, int *slurm_processes) {
    char *env_nodes = getenv("SLURM_JOB_NUM_NODES");
    char *env_threads = getenv("SLURM_CPUS_PER_TASK");
    char *env_processes = getenv("SLURM_NTASKS");
    *slurm_nodes = env_nodes ? atoi(env_nodes) : 0;
    *slurm_threads = env_threads ? atoi(env_threads) : 0;
    *slurm_processes = env_processes ? atoi(env_processes) : 0;
}


// Get system information from SLURM environment variables
void get_system_info(int *nodes, int *threads, int *processes) {
    // Read from SLURM environment variables
    char *env_val;
    
    // Number of nodes allocated
    env_val = getenv("SLURM_NNODES");
    *nodes = env_val ? atoi(env_val) : 1;
    
    // Number of threads (prefer OMP_NUM_THREADS, fallback to SLURM_CPUS_PER_TASK)
    env_val = getenv("OMP_NUM_THREADS");
    if (env_val) {
        *threads = atoi(env_val);
    } else {
        env_val = getenv("SLURM_CPUS_PER_TASK");
        *threads = env_val ? atoi(env_val) : 1;
    }
    
    // Number of MPI processes/tasks
    env_val = getenv("SLURM_NTASKS");
    *processes = env_val ? atoi(env_val) : 1;
}


// Get the next iteration number (thread-safe counter)
int get_next_iteration_number() {
    static int iteration_counter = 0;
    return iteration_counter++;
}

int main(int argc, char **argv) {
    // Initialize MPI 
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // MPI rank
    MPI_Comm_size(MPI_COMM_WORLD, &size); //MPI processes

    // Only rank 0 prints (prevents duplicate output with multiple MPI processes)
    if (rank == 0) {
        printf("Monte Carlo Financial Risk Simulation\n");
        if (size > 1) {
            printf("Running with %d MPI processes\n\n", size);
        } else {
            printf("\n");
        }
        printf("Loading parameters from binary files...\n");
    }

    // Load mu and Sigma and actual_freq from binary files
    gsl_vector *mu = load_mu_binary("data/mu.bin");
    gsl_matrix *Sigma = load_sigma_binary("data/sigma.bin");
    double actual_freq = load_actual_freq_binary("data/params.bin");

    if (!mu || !Sigma) {
        if (rank == 0) {
            fprintf(stderr, "Error: Failed to load binary files. Run Python preprocessing first.\n");
        }
        if (mu) gsl_vector_free(mu);
        if (Sigma) gsl_matrix_free(Sigma);
        MPI_Finalize(); // early exit
        return 1;
    }

    if (actual_freq < 0.0 && rank == 0) {
        fprintf(stderr, "Warning: Failed to load actual_freq from params.bin, using NaN\n");
        actual_freq = NAN;
    }


    int N = mu->size;  // Number of assets

    // Load configuration parameters from config.yaml
    ConfigParams config;
    if (load_config("config.yaml", &config) != 0) {
        if (rank == 0) {
            fprintf(stderr, "Error: Failed to load config.yaml. Exiting.\n");
        }
        gsl_vector_free(mu);
        gsl_matrix_free(Sigma);
        MPI_Finalize(); // early exit
        return 1;
    }

    int k = config.k;
    double x = config.x;
    int M = config.M;

    if (rank == 0) {
        printf("\nSimulation Configuration:\n");
        printf("  Number of assets (N): %d\n", N);
        printf("  Crash threshold (k): %d\n", k);
        printf("  Return threshold (x): %.2f%%\n", x * 100);
        printf("  Number of trials (M): %d\n", M);
        printf("  Training ratio: %.2f%%\n", config.train_ratio * 100);
    }

    // Get user comment from config 
    char *user_comment = config.comment;

    // Initialize parameters
    MonteCarloParams *params = (MonteCarloParams *)malloc(sizeof(MonteCarloParams));
    if (!params) {
        if (rank == 0) {
            fprintf(stderr, "Error: Failed to allocate parameters\n");
        }
        gsl_vector_free(mu);
        gsl_matrix_free(Sigma);
        free_config(&config);
        MPI_Finalize(); // early exit
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

    int slurm_nodes, slurm_threads, slurm_processes;
    get_slurm_info(&slurm_nodes, &slurm_threads, &slurm_processes);
    // Models to run - TODO: later read from config
    // const char *models_to_run[] = {"serial", "openmp"};
    // int num_models = sizeof(models_to_run) / sizeof(models_to_run[0])

    // Run all models from config
    for (int i = 0; i < config.num_models; i++) {
        const char *model_type = config.models[i];
        MonteCarloResult result = {0};

        // Explicitly initialize timing field to "not measured"
        result.kernel_time_ms = -1.0;

        ModelFunctions model;

        if (rank == 0) {
            printf("\n=== Running %s model ===\n", model_type);
        }

        // Select model based on type
        if (strcmp(model_type, "serial") == 0) {
#ifdef SERIAL_BUILD
            model = get_serial_model();
#else
            fprintf(stderr, "Error: Serial model not available in this build\n");
            continue;
#endif
        }
        else if (strcmp(model_type, "openmp") == 0) {
#ifdef OPENMP_BUILD
            model = get_openmp_model();
#else
            fprintf(stderr, "Error: OpenMP model not available in this build\n");
            continue;
#endif
        }
        else if(strcmp(model_type, "openmp_opt") == 0) {
#ifdef OPENMP_OPT_BUILD
            model = get_openmp_opt_model();
#else
            fprintf(stderr, "Error: OpenMP Optimized model not available in this build\n");
            continue;
#endif 
        }           
        else if (strcmp(model_type, "mpi_openmp") == 0) {
#ifdef MPI_OPENMP_BUILD
            model = get_mpi_openmp_model();
#else
            fprintf(stderr, "Error: MPI model not available in this build\n");
            continue;
#endif
        }
        else if (strcmp(model_type, "cuda") == 0 || strcmp(model_type, "gpu") == 0) {
#ifdef CUDA_BUILD
            model = get_cuda_model();
#else
            fprintf(stderr, "Error: CUDA model not available in this build\n");
            continue;
#endif
        }
        else {
            if (rank == 0) {
                fprintf(stderr, "Error: Unknown model type '%s'\n", model_type);
            }
            continue;
        }

        // Record start time (wall-clock time for accurate GPU timing)
        struct timespec start_time, end_time;
        clock_gettime(CLOCK_MONOTONIC, &start_time);

        // Run model (all ranks participate for MPI models, serial/OpenMP run normally)
        int status = model.run_model(params, &result);

        // Calculate execution time in milliseconds (wall-clock time)
        clock_gettime(CLOCK_MONOTONIC, &end_time);
        long execution_time_ms = (long)((end_time.tv_sec - start_time.tv_sec) * 1000L +
                                        (end_time.tv_nsec - start_time.tv_nsec) / 1000000L);

        // Get throughput in seconds
        int throughput = (int)round((double)M * 1000.0 / (double)execution_time_ms);

        // Calculate detailed timing metrics from kernel time
        double kernel_time_ms = result.kernel_time_ms;
        double overhead_time_ms = -1.0;
        double kernel_throughput = -1.0;

        if (kernel_time_ms >= 0.0) {
            // Model supports kernel timing - calculate overhead and throughput
            overhead_time_ms = (double)execution_time_ms - kernel_time_ms;

            // Calculate throughput: trials per second
            // Convert kernel_time_ms to seconds, then divide trials by time
            if (kernel_time_ms > 0.0) {
                double kernel_time_sec = kernel_time_ms / 1000.0;
                kernel_throughput = (double)M / kernel_time_sec;
            }
            // else: kernel_time_ms == 0.0, keep throughput as -1.0 (can't divide by zero)
        }

        if (status == 0) {
            // Only rank 0 prints results and writes to CSV
            if (rank == 0) {
                print_results(model.name, &result, M);

                // Write results to CSV
                SimulationResultsData results_data = {
                    .iteration_id = iteration_id,
                    .timestamp = (long)time(NULL),
                    .execution_time_ms = execution_time_ms,
                    .MC_throughput_secs = throughput,
                    .kernel_time_ms = kernel_time_ms,
                    .overhead_time_ms = overhead_time_ms,
                    .kernel_throughput = kernel_throughput,
                    .comment = user_comment ? user_comment : "",
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
            }
        } else {
            if (rank == 0) {
                fprintf(stderr, "Error: %s model failed\n", model.name);
            }
        }

        // Always cleanup S_values regardless of success/failure
        make_clean(&result);

        if (rank == 0) {
            printf("\n");
        }
    }

    free_params(params);
    free_config(&config);

    if (rank == 0) {
        printf("Simulation complete.\n");
    }

    MPI_Finalize();
    return 0;
}