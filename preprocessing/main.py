import numpy as np
import yaml
from pathlib import Path
from fetch_split import fetch_and_split
from preprocessing_stats import compute_stats
from evaluation import compute_metrics

def load_config(config_path="../config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def run_pipeline(config_path="../config.yaml"):
    """
    Execute the complete preprocessing and evaluation pipeline.
    Reads all parameters from YAML config file.
    """

    # Load configuration
    config = load_config(config_path)

    # Extract data parameters
    data_config = config['data']
    start = data_config['start']
    end = data_config['end']
    train_ratio = data_config['train_ratio']

    # Convert indices list of dicts to single dict
    INDICES = {}
    for idx_dict in data_config['indices']:
        INDICES.update(idx_dict)

    # Extract simulation parameters
    sim_config = config['simulation']
    x = sim_config['x']
    k = sim_config['k']
    N = sim_config['N']

    # Extract CSV config
    csv_config = config['csv']
    csv_dir = csv_config['data_dir']


    # Step 1: Fetch and split data
    train, test = fetch_and_split(indices=INDICES, start=start, end=end, train_ratio=train_ratio)

    # Step 2: Compute statistical parameters
    mu, Sigma = compute_stats(train)

    # Step 3: Compute evaluation metrics
    actual_freq = compute_metrics(test, x=x, k=k)

    # Step 4: Save mu and Sigma to CSV files
    save_parameters_to_csv(mu, Sigma, output_dir=csv_dir)

    # Summary
    print("PIPELINE COMPLETED")
    print(f"Training data shape: {train.shape}")
    print(f"Test data shape: {test.shape}")
    print(f"Statistical parameters: mu{mu.shape}, Sigma{Sigma.shape}")
    print(f"Actual crash probability: {actual_freq:.6f}")
    print(f"Parameters saved to: mu.csv, Sigma.csv")

    # Return all results
    return {
        "train": train,
        "test": test,
        "mu": mu,
        "Sigma": Sigma,
        "actual_freq": actual_freq
    }


def save_parameters_to_csv(mu, Sigma, output_dir="."):
    """
    Save mu and Sigma to CSV files.

    INPUT:
        mu: Mean returns vector (1D numpy array)
        Sigma: Covariance matrix (2D numpy array)
        output_dir: Directory to save CSV files (default: current directory)

    OUTPUT:
        Saves two CSV files:
        - mu.csv: Single column with N mean return values
        - Sigma.csv: NxN covariance matrix
    """
    # Save mu to CSV (single column)
    mu_path = f"{output_dir}/mu.csv"
    np.savetxt(mu_path, mu, fmt="%.6f")
    print(f"Saved mu to: {mu_path}")

    # Save Sigma to CSV (matrix)
    sigma_path = f"{output_dir}/Sigma.csv"
    np.savetxt(sigma_path, Sigma, fmt="%.6f", delimiter=",")
    print(f"Saved Sigma to: {sigma_path}")


if __name__ == "__main__":
    # Execute pipeline with config file
    results = run_pipeline(config_path="../config.yaml")

    mu = results["mu"]
    Sigma = results["Sigma"]
    test = results["test"]
    actual_freq = results["actual_freq"]

    print(f"\nResults:")
    n_show = min(5, len(mu))
    print(f"mu (first {n_show}): {mu[:n_show]}")
    print(f"Sigma shape: {Sigma.shape}")
    print(f"Sigma (top-left {n_show}x{n_show}):\n{Sigma[:n_show, :n_show]}")
    print(f"test shape: {test.shape}")
    print(f"actual_freq: {actual_freq:.6f}")