#ifndef LOAD_BINARY_H
#define LOAD_BINARY_H

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>

// Load mu (mean vector) from binary file
// File format: [N (int32)] + [mu values (double)]
gsl_vector* load_mu_binary(const char *filename);

// Load Sigma (covariance matrix) from binary file
// File format: [rows (int32)] + [cols (int32)] + [Sigma values row-major (double)]
gsl_matrix* load_sigma_binary(const char *filename);


#endif // LOAD_BINARY_H
