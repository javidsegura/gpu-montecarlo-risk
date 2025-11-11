#ifndef LOAD_CONFIG_H
#define LOAD_CONFIG_H

// Structure to hold indices with their ticker symbols
typedef struct {
    char *name;        // Index name (e.g., "STOXX50E")
    char *ticker;      // Ticker symbol (e.g., "^STOXX50E")
} IndexConfig;

// Structure to hold all configuration parameters from YAML
typedef struct {
    // Date range
    char *start;                       // Start date (YYYY-MM-DD)
    char *end;                         // End date (YYYY-MM-DD)

    // Simulation parameters
    double x;                          // Return threshold (e.g., 0.02 for 2%)
    int N;                             // Number of assets (from context, optional in YAML)
    int M;                             // Number of Monte Carlo trials
    int k;                             // Crash threshold (consecutive days)
    unsigned long random_seed;         // Random seed for reproducibility

    // Data parameters
    double train_ratio;                // Training data ratio (e.g., 0.8)

    // Indices configuration
    IndexConfig *indices;              // Array of indices
    int num_indices;                   // Number of indices

    // Models to run
    char **models;                     // Array of model names
    int num_models;                    // Number of models

    // Additional notes
    char *comment;                     // User comment/description
} ConfigParams;

// Load configuration from YAML file
// Returns 0 on success, -1 on failure
int load_config(const char *filename, ConfigParams *config);

// Free all dynamically allocated memory in ConfigParams
void free_config(ConfigParams *config);

#endif // LOAD_CONFIG_H
