/**
 * Configuration header - Provides config structure and parser
 *
 * Reads YAML config file and populates configuration structure.
 */

#ifndef CONFIG_H
#define CONFIG_H

#include <stdio.h>

typedef struct {
    int N;                  // Number of assets
    int k;                  // Crash threshold
    double x;               // Return threshold
    int M;                  // Number of trials
    char mu_file[256];      // Path to mu CSV
    char sigma_file[256];   // Path to Sigma CSV
    char comment[512];      // Config comment
} SimulationConfig;

/**
 * Load configuration from YAML file
 *
 * For now, uses simple parsing (no external YAML library).
 * Reads config.yaml and extracts:
 * - N, k, x, M from simulation section
 * - mu_file, sigma_file from csv section
 * - comment from results section
 *
 * Returns 0 on success, -1 on failure
 */
int load_config(const char *config_path, SimulationConfig *config);

#endif // CONFIG_H
