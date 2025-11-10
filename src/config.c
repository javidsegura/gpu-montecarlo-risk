/**
 * Configuration parser - Simple YAML parser for config.yaml
 *
 * Reads YAML config and populates SimulationConfig structure.
 * Uses basic string parsing (no external library).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "config.h"

#define MAX_LINE 1024

/**
 * Trim whitespace from string
 */
static void trim(char *str) {
    int start = 0, end = strlen(str) - 1;
    while (start <= end && (str[start] == ' ' || str[start] == '\t')) start++;
    while (end >= start && (str[end] == ' ' || str[end] == '\t' || str[end] == '\n' || str[end] == '\r')) end--;
    memmove(str, str + start, end - start + 1);
    str[end - start + 1] = '\0';
}

/**
 * Extract value from YAML line (e.g., "N: 9" -> "9")
 */
static char* extract_value(char *line) {
    char *colon = strchr(line, ':');
    if (!colon) return NULL;
    colon++;
    while (*colon == ' ' || *colon == '\t') colon++;
    // Remove quotes if present
    if (*colon == '"' || *colon == '\'') {
        colon++;
        char *end = strchr(colon, '"');
        if (!end) end = strchr(colon, '\'');
        if (end) *end = '\0';
    }
    // Remove trailing newline
    char *nl = strchr(colon, '\n');
    if (nl) *nl = '\0';
    return colon;
}

/**
 * Load configuration from YAML file
 */
int load_config(const char *config_path, SimulationConfig *config) {
    if (!config_path || !config) {
        fprintf(stderr, "Error: Invalid arguments to load_config\n");
        return -1;
    }

    // Initialize defaults
    config->N = 9;
    config->k = 5;
    config->x = 0.02;
    config->M = 100000;
    strcpy(config->mu_file, "mu.csv");
    strcpy(config->sigma_file, "Sigma.csv");
    strcpy(config->comment, "GPU Monte Carlo Risk Simulation");

    // Open config file
    FILE *f = fopen(config_path, "r");
    if (!f) {
        fprintf(stderr, "Error: Cannot open config file: %s\n", config_path);
        return -1;
    }

    char line[MAX_LINE];
    int in_simulation = 0, in_csv = 0, in_results = 0;

    while (fgets(line, sizeof(line), f)) {
        // Skip comments and empty lines
        if (line[0] == '#' || line[0] == '\n') continue;

        // Check for section headers
        if (strstr(line, "simulation:")) {
            in_simulation = 1;
            in_csv = 0;
            in_results = 0;
            continue;
        }
        if (strstr(line, "csv:")) {
            in_csv = 1;
            in_simulation = 0;
            in_results = 0;
            continue;
        }
        if (strstr(line, "results:")) {
            in_results = 1;
            in_csv = 0;
            in_simulation = 0;
            continue;
        }

        // Parse values based on section
        if (in_simulation) {
            if (strstr(line, "N:")) {
                char *val = extract_value(line);
                if (val) config->N = atoi(val);
            }
            else if (strstr(line, "k:")) {
                char *val = extract_value(line);
                if (val) config->k = atoi(val);
            }
            else if (strstr(line, "x:")) {
                char *val = extract_value(line);
                if (val) config->x = atof(val);
            }
            else if (strstr(line, "M:")) {
                char *val = extract_value(line);
                if (val) config->M = atoi(val);
            }
        }
        else if (in_csv) {
            if (strstr(line, "mu_file:")) {
                char *val = extract_value(line);
                if (val) {
                    strncpy(config->mu_file, val, sizeof(config->mu_file) - 1);
                    config->mu_file[sizeof(config->mu_file) - 1] = '\0';
                }
            }
            else if (strstr(line, "sigma_file:")) {
                char *val = extract_value(line);
                if (val) {
                    strncpy(config->sigma_file, val, sizeof(config->sigma_file) - 1);
                    config->sigma_file[sizeof(config->sigma_file) - 1] = '\0';
                }
            }
        }
        else if (in_results) {
            if (strstr(line, "comment:")) {
                char *val = extract_value(line);
                if (val) {
                    strncpy(config->comment, val, sizeof(config->comment) - 1);
                    config->comment[sizeof(config->comment) - 1] = '\0';
                }
            }
        }
    }

    fclose(f);

    // Validate configuration
    if (config->N <= 0 || config->k <= 0 || config->M <= 0 || config->x <= 0) {
        fprintf(stderr, "Error: Invalid configuration values\n");
        return -1;
    }

    return 0;
}
