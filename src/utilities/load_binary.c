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

    // Allocate matrix
    gsl_matrix *Sigma = gsl_matrix_alloc(rows, cols);
    if (!Sigma) {
        fprintf(stderr, "Error: Failed to allocate Sigma matrix\n");
        fclose(f);
        return NULL;
    }

    // Read data (row-major order from Python matches GSL default)
    if (fread(Sigma->data, sizeof(double), rows * cols, f) != (size_t)(rows * cols)) {
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

    fclose(f);
    printf("Loaded actual_freq (%.6f) from %s\n", actual_freq, filename);
    return actual_freq;
}
