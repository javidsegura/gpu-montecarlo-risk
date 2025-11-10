#!/usr/bin/env python3
"""
Generate test CSV files for validating CSV reader implementations.

Creates CSV files with mu and Sigma datasets for testing the C and Python readers.

TESTING STRATEGY & WHY WE USE LOCAL MOCKS:
===========================================

This test generation script creates mock CSV data locally for validation of our CSV reader
implementations (src/csv_io.c and preprocessing/load_csv.py).

IMPORTANT: We are doing local testing with mocks ONLY BECAUSE THE CLUSTER IS CURRENTLY DOWN.
Once the cluster is back online, we will transition to testing with real financial data on
the actual HPC infrastructure.

WHY LOCAL TESTING WITH MOCKS (Current Temporary Solution):
-----------------------------------------------------------
The cluster infrastructure is unavailable, so we:
1. Create mock CSV data with known dimensions locally
2. Validate reader logic works correctly on local machine
3. Test both success cases and error handling edge cases
4. Prepare code for cluster deployment when infrastructure is available

This is a workaround for the cluster outage.
Once cluster is operational, testing will move to real financial data and production hardware.

HOW TO RUN TESTS:
-----------------
1. Generate mock CSV files (from project root):
   python3 tests/generate_test_csv.py

2. Test C implementation:
   cd tests && gcc -o test_csv_io test_csv_io.c ../src/csv_io.c \
     -I/opt/homebrew/Cellar/gsl/2.8/include \
     -L/opt/homebrew/Cellar/gsl/2.8/lib -lgsl -lgslcblas -lm
   ./test_csv_io

3. Test Python implementation:
   cd tests && python3 test_load_csv.py

Expected Output:
  C tests: 5/5 PASS
  Python tests: 5/5 PASS

TESTING COVERAGE:
-----------------
Test 1: Valid data (mu.csv=9 values, Sigma.csv=9x9) → returns 0 (success)
Test 2: Missing mu file → returns -1 (file not found)
Test 3: Missing Sigma file → returns -1 (file not found)
Test 4: Wrong mu dimension (10 values instead of 9) → returns -3 (dimension mismatch)
Test 5: Wrong Sigma dimension (10x10 instead of 9x9) → returns -3 (dimension mismatch)

This validates that our readers:
- Handle valid inputs correctly
- Detect missing files appropriately
- Catch dimension mismatches (critical for financial data correctness)
- Prevent silent data corruption from malformed CSV files

WHEN TO MOVE TO CLUSTER TESTING:
--------------------------------
After local tests pass 100%, use cluster for:
- Real financial data (9 European stock indices over 10 years)
- Large-scale Monte Carlo simulations (M=1,000,000+ iterations)
- Performance benchmarking with actual HPC hardware
- Stress testing with realistic data volumes
"""

import numpy as np
import os

def create_valid_files():
    """Create valid CSV files with correct structure."""
    N = 9

    # Generate test data
    mu = np.random.randn(N) * 0.001
    Sigma = np.random.randn(N, N)
    Sigma = (Sigma + Sigma.T) / 2  # Make symmetric

    # Save to CSV
    np.savetxt("tests/mu.csv", mu, fmt="%.6f")
    np.savetxt("tests/Sigma.csv", Sigma, fmt="%.6f", delimiter=",")

    print(f"✓ Created valid CSV files")
    print(f"  mu shape: {mu.shape}")
    print(f"  Sigma shape: {Sigma.shape}")


def create_wrong_dim_mu():
    """Create CSV with wrong dimension for mu."""
    N = 10  # Wrong size instead of 9
    mu = np.ones(N) * 0.001

    np.savetxt("tests/mu_wrong_dim.csv", mu, fmt="%.6f")
    print(f"✓ Created mu_wrong_dim.csv (10 values instead of 9)")


def create_wrong_dim_sigma():
    """Create CSV with wrong dimension for Sigma."""
    N = 10  # Wrong size instead of 9
    Sigma = np.eye(N)

    np.savetxt("tests/Sigma_wrong_dim.csv", Sigma, fmt="%.6f", delimiter=",")
    print(f"✓ Created Sigma_wrong_dim.csv (10x10 instead of 9x9)")


if __name__ == "__main__":
    os.makedirs("tests", exist_ok=True)

    print("Generating test CSV files...\n")

    create_valid_files()
    create_wrong_dim_mu()
    create_wrong_dim_sigma()

    print("\n✓ All test CSV files created in tests/ directory")
