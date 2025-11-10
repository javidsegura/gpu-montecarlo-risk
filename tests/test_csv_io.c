/**
 * Test program for csv_io.c
 *
 * Tests the read_csv_parameters() function with various CSV files.
 *
 * COMPILE:
 *   gcc -o test_csv_io test_csv_io.c ../src/csv_io.c -lgsl -lgslcblas -lm
 *
 * RUN:
 *   ./test_csv_io
 */

#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include "../src/csv_io.h"

int run_test(const char *mu_file, const char *sigma_file, int N,
             int expected_return, const char *test_name) {
    gsl_vector *mu = NULL;
    gsl_matrix *Sigma = NULL;

    printf("\nTest: %s\n", test_name);
    printf("  Files: %s, %s\n", mu_file, sigma_file);
    printf("  Expected return: %d\n", expected_return);

    int ret = read_csv_parameters(mu_file, sigma_file, N, &mu, &Sigma);

    printf("  Actual return: %d\n", ret);

    if (ret == expected_return) {
        printf("  Result: ✓ PASS\n");

        if (ret == 0) {
            printf("  Data loaded:\n");
            printf("    mu[0] = %.6f\n", gsl_vector_get(mu, 0));
            printf("    Sigma[0,0] = %.6f\n", gsl_matrix_get(Sigma, 0, 0));

            gsl_vector_free(mu);
            gsl_matrix_free(Sigma);
        }
        return 1;
    } else {
        printf("  Result: ✗ FAIL (expected %d, got %d)\n", expected_return, ret);
        return 0;
    }
}

int main(void) {
    printf("========================================\n");
    printf("CSV I/O Module Test Suite\n");
    printf("========================================\n");

    int N = 9;
    int passed = 0;
    int total = 0;

    // Test 1: Valid files
    total++;
    if (run_test("mu.csv", "Sigma.csv", N, 0,
                 "Test 1: Load valid CSV files")) {
        passed++;
    }

    // Test 2: Missing mu file
    total++;
    if (run_test("nonexistent_mu.csv", "Sigma.csv", N, -1,
                 "Test 2: Missing mu file")) {
        passed++;
    }

    // Test 3: Missing Sigma file
    total++;
    if (run_test("mu.csv", "nonexistent_Sigma.csv", N, -1,
                 "Test 3: Missing Sigma file")) {
        passed++;
    }

    // Test 4: Wrong dimension for mu
    total++;
    if (run_test("mu_wrong_dim.csv", "Sigma.csv", N, -3,
                 "Test 4: Dimension mismatch in mu")) {
        passed++;
    }

    // Test 5: Wrong dimension for Sigma
    total++;
    if (run_test("mu.csv", "Sigma_wrong_dim.csv", N, -3,
                 "Test 5: Dimension mismatch in Sigma")) {
        passed++;
    }

    // Print summary
    printf("\n========================================\n");
    printf("Test Summary\n");
    printf("========================================\n");
    printf("Passed: %d/%d\n", passed, total);

    if (passed == total) {
        printf("Result: ✓ ALL TESTS PASSED\n");
        return 0;
    } else {
        printf("Result: ✗ SOME TESTS FAILED\n");
        return 1;
    }
}
