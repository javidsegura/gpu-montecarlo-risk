#ifndef CSV_WRITER_H
#define CSV_WRITER_H

#include "load_config.h"

// Structure to hold all simulation results data
typedef struct {
    // Metadata
    int iteration_id;                  // Unique identifier for this simulation run
    long timestamp;                    // Unix timestamp when simulation was run
    long execution_time_ms;            // Execution time in milliseconds
    int MC_throughput_secs;            // Monte Carlo throughput in seconds
    const char *comment;               // User comment

    // Configuration parameters
    const char *start_date;            // Start date for historical data
    const char *end_date;              // End date for historical data
    double train_ratio;                // Training data ratio
    int M;                             // Number of Monte Carlo trials
    int k;                             // Crash threshold
    double x;                          // Return threshold

    // Model and system information
    const char *model_name;            // Name of the model used
    unsigned long seed;                // Random seed used for RNG
    int nodes;                         // Number of nodes 
    int threads;                       // Number of threads 
    int processes;                     // Number of processes 


    // Index information
    IndexConfig *indices;              // Array of indices used
    int num_indices;                   // Number of indices
    double actual_freq;                // Actual frequency from params.bin

    // Simulation results
    double P_hat;                      // Estimated probability of extreme event
    int count;                         // Count of extreme events
    double std_error;                  // Standard error
    double ci_lower;                   // Lower confidence interval
    double ci_upper;                   // Upper confidence interval
} SimulationResultsData;


// Write simulation results to CSV file
// Thread-safe: uses file locking to prevent concurrent write issues
int write_results_to_csv(const char *filepath, const SimulationResultsData *data);

#endif // CSV_WRITER_H
