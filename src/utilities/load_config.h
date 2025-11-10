#ifndef LOAD_CONFIG_H
#define LOAD_CONFIG_H

// Structure to hold configuration parameters
typedef struct {
    int M;        // Number of Monte Carlo trials
    double x;     // Return threshold (e.g., 0.02 for 2%)
    int k;        // Crash threshold (consecutive days)
} ConfigParams;

// Load configuration from YAML file
// Returns 0 on success, -1 on failure
int load_config(const char *filename, ConfigParams *config);

#endif // LOAD_CONFIG_H
