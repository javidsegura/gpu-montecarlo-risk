#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_blas.h>

typedef struct {
    double P_hat;
    int count;
    double *S_values;
} SimulationResult;

/**
 * Monte Carlo crash simulation
 * 
 * Parameters:
 * - x: Drop percentage threshold
 * - N: Number of indices
 * - k: Number of crashes needed for extreme event
 * - mu: Mean return vector (length N)
 * - Sigma: Covariance matrix (N x N)
 * - M: Number of Monte Carlo trials
 * 
 * Returns: SimulationResult structure
 */
SimulationResult monte_carlo_crash_simulation(double x, int N, int k, 
                                             const double *mu, 
                                             const gsl_matrix *Sigma, 
                                             int M) {
    SimulationResult result;
    result.S_values = (double *)malloc(M * sizeof(double));
    result.count = 0;
    
    // Allocate matrices and vectors
    gsl_matrix *L = gsl_matrix_alloc(N, N);
    gsl_vector *Z = gsl_vector_alloc(N);
    gsl_vector *R = gsl_vector_alloc(N);
    gsl_vector *mu_vec = gsl_vector_alloc(N);
    
    // Copy mu to gsl_vector
    for (int i = 0; i < N; i++) {
        gsl_vector_set(mu_vec, i, mu[i]);
    }
    
    // BEFORE SIMULATION: Cholesky decomposition
    // Copy Sigma to L (gsl_linalg_cholesky_decomp modifies in place)
    gsl_matrix_memcpy(L, Sigma);
    gsl_linalg_cholesky_decomp1(L);
    
    printf("Starting Monte Carlo simulation with M = %d trials...\n", M);
    printf("Parameters: N=%d, k=%d, x=%.1f%%\n", N, k, x * 100);
    printf("------------------------------------------------------------\n");
    
    // Setup random number generator
    gsl_rng *rng = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(rng, 42); // Set seed for reproducibility
    
    // SIMULATION LOOP
    for (int j = 0; j < M; j++) {
        // (i) Generate vector of independent N(0,1) variables
        for (int i = 0; i < N; i++) {
            gsl_vector_set(Z, i, gsl_ran_gaussian(rng, 1.0));
        }
        
        // R = μ + L*Z
        // First: R = L*Z
        gsl_blas_dgemv(CblasNoTrans, 1.0, L, Z, 0.0, R);
        // Then: R = R + μ
        gsl_vector_add(R, mu_vec);
        
        // (ii) Compute S^(j) = sum of I(R_i^(j) < -x)
        int S = 0;
        for (int i = 0; i < N; i++) {
            if (gsl_vector_get(R, i) < -x) {
                S++;
            }
        }
        result.S_values[j] = (double)S;
        
        // (iii) Check if S^(j) >= k
        if (S >= k) {
            result.count++;
        }
        
        // Progress indicator (every 10%)
        if ((j + 1) % (M / 10) == 0) {
            printf("Progress: %d%% - Current estimate: %.6f\n", 
                   (j + 1) * 100 / M, (double)result.count / (j + 1));
        }
    }
    
    // Compute probability: P̂_MIN = C/M
    result.P_hat = (double)result.count / M;
    
    printf("------------------------------------------------------------\n");
    printf("Simulation complete!\n");
    
    // Cleanup
    gsl_rng_free(rng);
    gsl_matrix_free(L);
    gsl_vector_free(Z);
    gsl_vector_free(R);
    gsl_vector_free(mu_vec);
    
    return result;
}

/**
 * Compute accuracy and confidence interval
 */
void compute_accuracy(double P_hat, int M, double *std_error, 
                     double *ci_lower, double *ci_upper) {
    // Standard error: sqrt(p(1-p)/M)
    *std_error = sqrt(P_hat * (1.0 - P_hat) / M);
    
    double z_score = 1.96; // for 95% confidence
    double margin = z_score * (*std_error);
    
    *ci_lower = fmax(0.0, P_hat - margin);
    *ci_upper = fmin(1.0, P_hat + margin);
}

int main() {
    // =============================================================================
    // EXECUTION EXAMPLE
    // =============================================================================
    
    // Define parameters
    int N = 10;          // Number of indices
    int k = 5;           // At least 5 must crash
    double x = 0.02;     // 2% drop threshold
    int M = 100000;      // Number of Monte Carlo trials
    
    // Define mean returns (assume zero for simplicity)
    double *mu = (double *)calloc(N, sizeof(double));
    
    // Define covariance matrix (with some correlation)
    double rho = 0.3;       // Correlation coefficient
    double variance = 0.04; // 4% variance (2% std dev)
    
    gsl_matrix *Sigma = gsl_matrix_alloc(N, N);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i == j) {
                gsl_matrix_set(Sigma, i, j, variance);
            } else {
                gsl_matrix_set(Sigma, i, j, rho * variance);
            }
        }
    }
    
    // Run simulation
    SimulationResult result = monte_carlo_crash_simulation(x, N, k, mu, Sigma, M);
    
    // Compute accuracy
    double std_error, ci_lower, ci_upper;
    compute_accuracy(result.P_hat, M, &std_error, &ci_lower, &ci_upper);
    
    // Print results
    printf("\n============================================================\n");
    printf("FINAL RESULTS\n");
    printf("============================================================\n");
    printf("Estimated Probability (P̂_MIN): %.6f\n", result.P_hat);
    printf("Extreme Events Count: %d out of %d trials\n", result.count, M);
    printf("Standard Error: %.6f\n", std_error);
    printf("95%% Confidence Interval: [%.6f, %.6f]\n", ci_lower, ci_upper);
    printf("============================================================\n");
    
    // Cleanup
    free(mu);
    free(result.S_values);
    gsl_matrix_free(Sigma);
    
    return 0;
}