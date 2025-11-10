import numpy as np
from pathlib import Path


def save_results_binary(mu, Sigma, actual_freq, output_dir):
    """
    Save preprocessing results in binary format for C consumption.

    File format:
    - mu.bin: [N (int32)] + [mu values (double)]
    - sigma.bin: [N (int32)] + [N (int32)] + [Sigma values row-major (double)]
    - params.bin: [actual_freq (double)]

    Args:
        mu: Mean vector (numpy array)
        Sigma: Covariance matrix (numpy array)
        actual_freq: Actual crash frequency (float)
        output_dir: Directory to save files (Path or str)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save mu (mean vector)
    mu_file = output_dir / "mu.bin"
    with open(mu_file, 'wb') as f:
        # Write dimension (N)
        np.array([len(mu)], dtype=np.int32).tofile(f)
        # Write data as double precision
        mu.astype(np.float64).tofile(f)
    print(f"Saved mu to {mu_file}")

    # Save Sigma (covariance matrix)
    sigma_file = output_dir / "sigma.bin"
    with open(sigma_file, 'wb') as f:
        # Write dimensions (rows, cols)
        np.array([Sigma.shape[0], Sigma.shape[1]], dtype=np.int32).tofile(f)
        # Write data in row-major order (C convention)
        Sigma.astype(np.float64).tofile(f)
    print(f"Saved Sigma to {sigma_file}")

    # Save additional parameters
    params_file = output_dir / "params.bin"
    with open(params_file, 'wb') as f:
        # Write actual frequency as double
        np.array([actual_freq], dtype=np.float64).tofile(f)
    print(f"Saved actual frequency to {params_file}")
