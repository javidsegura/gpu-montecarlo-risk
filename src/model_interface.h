#ifndef MODEL_INTERFACE_H
#define MODEL_INTERFACE_H

#include <gsl/gsl_rng.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>

// Holds the final results of the Monte Carlo simulation
typedef struct {
    double P_hat;           // Estimated probability of extreme event
    int count;              // Number of trials that exceeded threshold k
    int *S_values;          // Array storing crash count S for each trial
    double std_error;       // Standard error of the estimate
    double ci_lower;        // Lower bound of 95% confidence interval
    double ci_upper;        // Upper bound of 95% confidence interval

    // Performance metrics
    double kernel_time_ms;  // Time spent in computation kernel (-1.0 if not measured)
} MonteCarloResult;

// Input parameters for the Monte Carlo simulation
typedef struct {
    double x;               // Crash threshold (0.02 = 2%)
    int N;                  // Number of assets
    int k;                  // Minimum number of crashes to trigger extreme event
    gsl_vector *mu;         // Mean returns of assets
    gsl_matrix *Sigma;      // Covariance matrix of asset returns
    int M;                  // Number of Monte Carlo trials to run
    unsigned long random_seed;  // Random seed for reproducibility
} MonteCarloParams;

// Function pointer types - common interface that all models must implement
typedef int (*ModelRunFunc)(MonteCarloParams *params, MonteCarloResult *result);

// ModelFunctions: groups together function pointers for a given model implementation
typedef struct {
    const char *name;               // Model name ("serial", "openmp")
    ModelRunFunc run_model;         // Main simulation function with model-specific parallelization
} ModelFunctions;

#endif // MODEL_INTERFACE_H