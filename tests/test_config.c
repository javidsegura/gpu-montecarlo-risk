/**
 * Test program for config.c - Verifies YAML config parsing
 *
 * COMPILE:
 *   gcc -o test_config test_config.c ../src/config.c
 *
 * RUN:
 *   cd tests && ./test_config
 */

#include <stdio.h>
#include <string.h>
#include "../src/config.h"

int main() {
    printf("========================================\n");
    printf("Configuration Parser Test\n");
    printf("========================================\n\n");

    SimulationConfig config;
    int ret = load_config("../config.yaml", &config);

    if (ret != 0) {
        printf("✗ FAIL: Failed to load config\n");
        return 1;
    }

    printf("✓ Config loaded successfully\n\n");

    // Verify values
    int all_correct = 1;

    printf("Configuration Values:\n");
    printf("  N = %d (expected 9)\n", config.N);
    if (config.N != 9) {
        printf("    ✗ MISMATCH\n");
        all_correct = 0;
    }

    printf("  k = %d (expected 5)\n", config.k);
    if (config.k != 5) {
        printf("    ✗ MISMATCH\n");
        all_correct = 0;
    }

    printf("  x = %.4f (expected 0.0200)\n", config.x);
    if (config.x < 0.019 || config.x > 0.021) {
        printf("    ✗ MISMATCH\n");
        all_correct = 0;
    }

    printf("  M = %d (expected 100000)\n", config.M);
    if (config.M != 100000) {
        printf("    ✗ MISMATCH\n");
        all_correct = 0;
    }

    printf("  mu_file = '%s' (expected 'mu.csv')\n", config.mu_file);
    if (strcmp(config.mu_file, "mu.csv") != 0) {
        printf("    ✗ MISMATCH\n");
        all_correct = 0;
    }

    printf("  sigma_file = '%s' (expected 'Sigma.csv')\n", config.sigma_file);
    if (strcmp(config.sigma_file, "Sigma.csv") != 0) {
        printf("    ✗ MISMATCH\n");
        all_correct = 0;
    }

    printf("  comment = '%s'\n", config.comment);

    printf("\n========================================\n");
    if (all_correct) {
        printf("Result: ✓ ALL TESTS PASSED\n");
        return 0;
    } else {
        printf("Result: ✗ SOME TESTS FAILED\n");
        return 1;
    }
}
