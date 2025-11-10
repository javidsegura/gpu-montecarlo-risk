# C Serial Implementation

This is the baseline single-threaded C implementation used to measure speedup from parallel implementations. The serial version generates random numbers sequentially, ensuring fully reproducible results across runs with the same seed.

## Overview

The serial implementation computes Monte Carlo simulations for financial crash probability using the GNU Scientific Library. It follows a straightforward algorithm: generate correlated asset returns using Cholesky decomposition and count how many trials exceed the crash threshold.

## Data Structures

**MonteCarloParams** - Input parameters including N assets, k crash threshold, covariance matrix Sigma, mean returns mu, threshold x, and M trials.

**MonteCarloResult** - Output results containing probability estimate P_hat, count of extreme events, S_values array storing crashes per trial, standard error, and 95% confidence interval bounds.

**SerialModelState** - Internal state maintaining the Cholesky decomposition L matrix and a single random number generator instance.

## Core Functions

**serial_init()** - Preprocessing that computes the Cholesky decomposition once and initializes the random number generator with seed 42. The decomposition is reused across all trials for efficiency.

**serial_run_trial()** - Executes a single Monte Carlo trial by generating N independent standard normals, transforming them to correlated returns using R = mu + L*Z, and counting how many returns fall below the crash threshold.

**serial_simulate()** - Main simulation driver that allocates result arrays, initializes the model state, executes M independent trials sequentially, and calculates the probability estimate with confidence intervals.

**serial_cleanup_state()** - Releases memory for the random number generator and Cholesky matrix after simulation completes.


## Technical Details

The implementation uses the MT19937 Mersenne Twister random number generator with a fixed seed of 42, guaranteeing reproducible results across runs. The Cholesky decomposition is computed once during initialization and reused for all M trials to avoid redundant computation. Standard error is calculated as sqrt(P_hat * (1 - P_hat) / M) and the 95% confidence interval uses the normal approximation: P_hat ± 1.96 * standard error.
