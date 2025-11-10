#ifndef CSV_IO_H
#define CSV_IO_H

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>

/**
 * Load mu (mean vector) and Sigma (covariance matrix) from CSV files.
 *
 * INPUT:
 *   mu_filename: Path to CSV file containing mu (1D array)
 *   sigma_filename: Path to CSV file containing Sigma (2D matrix)
 *   N: Expected dimension (number of assets)
 *
 * OUTPUT:
 *   mu: GSL vector of size N (allocated by function)
 *   Sigma: GSL matrix of size NxN (allocated by function)
 *
 * RETURN:
 *   0 on success
 *   -1 on file not found
 *   -2 on invalid CSV format
 *   -3 on dimension mismatch
 *   -4 on memory allocation error
 *
 * ERROR HANDLING:
 *   - Validates both files exist before attempting to read
 *   - Validates CSV format is valid (proper number of values)
 *   - Validates mu has length N
 *   - Validates Sigma has dimensions NxN
 *   - Prints descriptive error messages to stderr
 *   - Caller must free allocated mu and Sigma on success
 */
int read_csv_parameters(const char *mu_filename, const char *sigma_filename, int N,
                        gsl_vector **mu, gsl_matrix **Sigma);

#endif // CSV_IO_H
