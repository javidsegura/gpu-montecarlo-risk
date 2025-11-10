"""
CSV Parameter Loader for Monte Carlo Simulation

This module provides functionality to read statistical parameters (mu and Sigma)
from CSV files with comprehensive validation and error handling.

CSV Format:
    mu.csv: Single column with N values (one per line)
    Sigma.csv: NxN matrix (comma or space separated values)
"""

import numpy as np
from pathlib import Path


def read_csv_parameters(mu_filename, sigma_filename, N):
    """
    Read mu (mean vector) and Sigma (covariance matrix) from CSV files.

    This function loads pre-computed statistical parameters from CSV files
    and validates that dimensions match the expected number of indices (N).

    Parameters
    ----------
    mu_filename : str or Path
        Path to CSV file containing mu (mean returns).
        Format: Single column with N values, one per line.
        Example: 'mu.csv'

    sigma_filename : str or Path
        Path to CSV file containing Sigma (covariance matrix).
        Format: NxN matrix, comma or space separated values.
        Example: 'Sigma.csv'

    N : int
        Expected number of financial indices (dimensions).
        Must be positive integer.
        mu should have N values and Sigma should have shape (N, N).

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray)
        - mu : Mean return vector of shape (N,)
        - Sigma : Covariance matrix of shape (N, N)
        Both as numpy arrays with dtype float64.

    Raises
    ------
    FileNotFoundError
        If either CSV file does not exist at the specified path.

    ValueError
        If dimensions of loaded data do not match expected dimensions.
        Includes specific messages about which dimension failed validation.

    RuntimeError
        If files cannot be read or are in invalid format.

    Examples
    --------
    >>> # Load parameters for 9 European stock indices
    >>> mu, Sigma = read_csv_parameters('mu.csv', 'Sigma.csv', N=9)
    >>> print(f"Mean returns shape: {mu.shape}")
    Mean returns shape: (9,)
    >>> print(f"Covariance matrix shape: {Sigma.shape}")
    Covariance matrix shape: (9, 9)

    >>> # Using with Monte Carlo simulation
    >>> from python_demo import monte_carlo_crash_simulation
    >>> P_hat, count, S_values = monte_carlo_crash_simulation(
    ...     x=0.02, N=9, k=5, mu=mu, Sigma=Sigma, M=100000
    ... )
    """

    mu_path = Path(mu_filename)
    sigma_path = Path(sigma_filename)

    # Check if mu file exists
    if not mu_path.exists():
        raise FileNotFoundError(
            f"mu CSV file not found: {mu_path}\n"
            f"Expected file path: {mu_path.absolute()}"
        )

    # Check if Sigma file exists
    if not sigma_path.exists():
        raise FileNotFoundError(
            f"Sigma CSV file not found: {sigma_path}\n"
            f"Expected file path: {sigma_path.absolute()}"
        )

    try:
        # Load mu from CSV (single column)
        mu = np.loadtxt(mu_filename, dtype=np.float64)

    except (ValueError, OSError) as e:
        raise RuntimeError(
            f"Failed to read mu CSV file: {mu_filename}\n"
            f"Error: {str(e)}"
        )

    try:
        # Load Sigma from CSV (2D matrix, comma-separated)
        Sigma = np.loadtxt(sigma_filename, dtype=np.float64, delimiter=',')

    except (ValueError, OSError) as e:
        raise RuntimeError(
            f"Failed to read Sigma CSV file: {sigma_filename}\n"
            f"Error: {str(e)}"
        )

    # Validate mu dimensions
    if mu.ndim != 1:
        raise ValueError(
            f"Invalid mu dimensions: expected 1D array, got {mu.ndim}D array.\n"
            f"mu shape: {mu.shape}"
        )

    if mu.shape[0] != N:
        raise ValueError(
            f"mu dimension mismatch: expected {N} values, got {mu.shape[0]}.\n"
            f"Parameter N={N} does not match number of values in mu"
        )

    # Validate Sigma dimensions
    if Sigma.ndim != 2:
        raise ValueError(
            f"Invalid Sigma dimensions: expected 2D array, got {Sigma.ndim}D array.\n"
            f"Sigma shape: {Sigma.shape}"
        )

    if Sigma.shape != (N, N):
        raise ValueError(
            f"Sigma dimension mismatch: expected shape ({N}, {N}), got shape {Sigma.shape}.\n"
            f"Parameter N={N} does not match Sigma.shape=({Sigma.shape[0]}, {Sigma.shape[1]})"
        )

    return mu, Sigma
