/**
 * CSV I/O Module - Read mu and Sigma from CSV files
 *
 * Provides functionality to load pre-computed financial parameters (mu and Sigma)
 * from CSV files with comprehensive validation and error handling.
 *
 * CSV Format:
 *   mu.csv: Single column with N values (one per line)
 *   Sigma.csv: NxN matrix (comma or space separated values)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include "csv_io.h"

/**
 * Read 1D CSV file into GSL vector
 * Returns number of values read, or -1 on error
 * Returns -2 if file contains more values than expected
 */
static int read_csv_vector(const char *filename, gsl_vector *v) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error: Cannot open file: %s\n", filename);
        return -1;
    }

    int count = 0;
    double value;
    int max_values = v->size;

    while (fscanf(file, "%lf", &value) == 1) {
        if (count >= max_values) {
            // File has more values than expected
            fclose(file);
            return -2;
        }
        gsl_vector_set(v, count, value);
        count++;
    }

    fclose(file);
    return count;
}

/**
 * Read 2D CSV file into GSL matrix
 * Returns number of rows read, or -1 on error
 * Returns -2 if file contains more rows or columns than expected
 */
static int read_csv_matrix(const char *filename, gsl_matrix *m) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error: Cannot open file: %s\n", filename);
        return -1;
    }

    int row = 0;
    int max_rows = m->size1;
    int max_cols = m->size2;
    double value;
    char line[4096];

    while (fgets(line, sizeof(line), file)) {
        if (row >= max_rows) {
            // File has more rows than expected
            fclose(file);
            return -2;
        }

        int col = 0;
        char *ptr = line;

        while (sscanf(ptr, "%lf", &value) == 1) {
            if (col >= max_cols) {
                // Row has more columns than expected
                fclose(file);
                return -2;
            }

            gsl_matrix_set(m, row, col, value);
            col++;

            // Skip past the number we just read
            while (*ptr && (*ptr == '-' || (*ptr >= '0' && *ptr <= '9') || *ptr == '.')) {
                ptr++;
            }
            while (*ptr && (*ptr == ',' || *ptr == ' ' || *ptr == '\t')) {
                ptr++;
            }
        }

        // Only count complete rows
        if (col == max_cols) {
            row++;
        } else if (col > 0) {
            // Incomplete row - too few columns
            fclose(file);
            return -2;
        }
    }

    fclose(file);
    return row;
}

/**
 * read_csv_parameters: Load mu and Sigma from CSV files with validation
 *
 * INPUT:
 *   mu_filename: Path to CSV file with mu values
 *   sigma_filename: Path to CSV file with Sigma matrix
 *   N: Expected dimension
 *
 * OUTPUT:
 *   mu: Allocated GSL vector of size N
 *   Sigma: Allocated GSL matrix of size NxN
 *
 * RETURN:
 *   0 on success
 *   -1 file not found
 *   -2 invalid CSV format
 *   -3 dimension mismatch
 *   -4 memory allocation error
 */
int read_csv_parameters(const char *mu_filename, const char *sigma_filename, int N,
                        gsl_vector **mu, gsl_matrix **Sigma) {
    int ret_code = 0;

    // Initialize output pointers
    *mu = NULL;
    *Sigma = NULL;

    // Validate input parameters
    if (mu_filename == NULL || sigma_filename == NULL || N <= 0) {
        fprintf(stderr, "Error: Invalid input parameters\n");
        return -1;
    }

    // Check if mu file exists
    if (access(mu_filename, F_OK) == -1) {
        fprintf(stderr, "Error: mu CSV file not found: %s\n", mu_filename);
        return -1;
    }

    // Check if Sigma file exists
    if (access(sigma_filename, F_OK) == -1) {
        fprintf(stderr, "Error: Sigma CSV file not found: %s\n", sigma_filename);
        return -1;
    }

    // Allocate GSL vector for mu
    *mu = gsl_vector_alloc(N);
    if (*mu == NULL) {
        fprintf(stderr, "Error: Failed to allocate mu vector\n");
        return -4;
    }

    // Allocate GSL matrix for Sigma
    *Sigma = gsl_matrix_alloc(N, N);
    if (*Sigma == NULL) {
        fprintf(stderr, "Error: Failed to allocate Sigma matrix\n");
        gsl_vector_free(*mu);
        *mu = NULL;
        return -4;
    }

    // Read mu from CSV
    int mu_count = read_csv_vector(mu_filename, *mu);
    if (mu_count < 0) {
        if (mu_count == -2) {
            // File has wrong number of values - dimension mismatch
            fprintf(stderr, "Error: mu dimension mismatch (file has different number of values than expected %d)\n", N);
            ret_code = -3;
        } else {
            ret_code = -2;
        }
        goto cleanup;
    }

    // Validate mu count
    if (mu_count != N) {
        fprintf(stderr, "Error: mu dimension mismatch (expected %d values, got %d)\n", N, mu_count);
        ret_code = -3;
        goto cleanup;
    }

    // Read Sigma from CSV
    int sigma_rows = read_csv_matrix(sigma_filename, *Sigma);
    if (sigma_rows < 0) {
        if (sigma_rows == -2) {
            // File has wrong dimensions - dimension mismatch
            fprintf(stderr, "Error: Sigma dimension mismatch (file has different dimensions than expected %dx%d)\n", N, N);
            ret_code = -3;
        } else {
            ret_code = -2;
        }
        goto cleanup;
    }

    // Validate Sigma dimensions
    if (sigma_rows != N) {
        fprintf(stderr, "Error: Sigma dimension mismatch (expected %dx%d matrix, got %dx%d)\n",
                N, N, sigma_rows, N);
        ret_code = -3;
        goto cleanup;
    }

    return 0;

cleanup:
    // Error path - free allocated GSL objects
    if (*mu != NULL) {
        gsl_vector_free(*mu);
        *mu = NULL;
    }
    if (*Sigma != NULL) {
        gsl_matrix_free(*Sigma);
        *Sigma = NULL;
    }

    return ret_code;
}
