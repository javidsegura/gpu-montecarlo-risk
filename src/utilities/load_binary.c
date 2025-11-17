#include "load_binary.h"
#include <stdio.h>
#include <stdlib.h>

gsl_vector* load_mu_binary(const char *filename) {
    FILE *f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Error: Cannot open %s\n", filename);
        return NULL;
    }

    // Read dimension
    int N;
    if (fread(&N, sizeof(int), 1, f) != 1) {
        fprintf(stderr, "Error: Failed to read dimension from %s\n", filename);
        fclose(f);
        return NULL;
    }

    // Validate dimension
    if (N <= 0 || N > 1000000) {  // Reasonable upper bound of 1M elements
        fprintf(stderr, "Error: Invalid dimension N=%d in %s (must be between 1 and 1000000)\n", N, filename);
        fclose(f);
        return NULL;
    }

    // Allocate vector
    gsl_vector *mu = gsl_vector_alloc(N);
    if (!mu) {
        fprintf(stderr, "Error: Failed to allocate mu vector\n");
        fclose(f);
        return NULL;
    }

    // Read data
    if (fread(mu->data, sizeof(double), N, f) != (size_t)N) {
        fprintf(stderr, "Error: Failed to read data from %s\n", filename);
        gsl_vector_free(mu);
        fclose(f);
        return NULL;
    }

    fclose(f);
    printf("Loaded mu (N=%d) from %s\n", N, filename);
    return mu;
}

gsl_matrix* load_sigma_binary(const char *filename) {
    FILE *f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Error: Cannot open %s\n", filename);
        return NULL;
    }

    // Read dimensions
    int rows, cols;
    if (fread(&rows, sizeof(int), 1, f) != 1 ||
        fread(&cols, sizeof(int), 1, f) != 1) {
        fprintf(stderr, "Error: Failed to read dimensions from %s\n", filename);
        fclose(f);
        return NULL;
    }

    // Validate dimensions
    if (rows <= 0 || cols <= 0) {
        fprintf(stderr, "Error: Invalid dimensions rows=%d, cols=%d in %s (must be positive)\n", rows, cols, filename);
        fclose(f);
        return NULL;
    }
    // Check for potential overflow in rows * cols (must fit in size_t safely)
    if ((long long)rows * (long long)cols > 2000000000LL) {  // Max 2B elements (< INT_MAX)
        fprintf(stderr, "Error: Matrix too large (%lld elements) in %s\n", (long long)rows * (long long)cols, filename);
        fclose(f);
        return NULL;
    }

    // Allocate matrix
    gsl_matrix *Sigma = gsl_matrix_alloc(rows, cols);
    if (!Sigma) {
        fprintf(stderr, "Error: Failed to allocate Sigma matrix\n");
        fclose(f);
        return NULL;
    }

    // Read data (row-major order from Python matches GSL default)
    size_t total_elements = (size_t)rows * (size_t)cols;
    if (fread(Sigma->data, sizeof(double), total_elements, f) != total_elements) {
        fprintf(stderr, "Error: Failed to read data from %s\n", filename);
        gsl_matrix_free(Sigma);
        fclose(f);
        return NULL;
    }

    fclose(f);
    printf("Loaded Sigma (%dx%d) from %s\n", rows, cols, filename);
    return Sigma;
}

double load_actual_freq_binary(const char *filename) {
    FILE *f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Error: Cannot open %s\n", filename);
        return -1.0;  // Return error
    }

    // Read actual frequency (single double value)
    double actual_freq;
    if (fread(&actual_freq, sizeof(double), 1, f) != 1) {
        fprintf(stderr, "Error: Failed to read actual_freq from %s\n", filename);
        fclose(f);
        return -1.0;  // Return error
    }

    // Validate frequency value
    if (actual_freq < 0.0 || actual_freq > 1.0) {
        fprintf(stderr, "Error: Invalid actual_freq=%.6f in %s (must be between 0.0 and 1.0)\n", actual_freq, filename);
        fclose(f);
        return -1.0;
    }

    fclose(f);
    printf("Loaded actual_freq (%.6f) from %s\n", actual_freq, filename);
    return actual_freq;
}
