import yaml
from pathlib import Path
from fetch_split import fetch_and_split
from preprocessing_stats import compute_stats
from evaluation import compute_metrics
from save_binary import save_results_binary

def run_pipeline():
    """
    Execute the complete preprocessing and evaluation pipeline.
    """
    config_file = Path(__file__).parent.parent / "config.yaml"
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    start = config['start']
    end = config['end']
    x = config['x']
    k = config['k']
    train_ratio = config['train_ratio']
    indices = config['indices']

    # Validate that indices is a dictionary
    if not isinstance(indices, dict):
        raise TypeError(
            f"'indices' in configs/config.yaml must be a dictionary mapping index names to ticker symbols, "
            f"got {type(indices).__name__} instead"
        )

    # Step 1: Fetch and split data
    train, test = fetch_and_split(indices=indices, start=start, end=end, train_ratio=train_ratio)

    # Step 2: Compute statistical parameters
    mu, Sigma = compute_stats(train)

    # Step 3: Compute evaluation metrics
    actual_freq = compute_metrics(test, x=x, k=k)

    # Step 4: Save results to binary files for C consumption
    output_dir = Path(__file__).parent.parent / "data"
    save_results_binary(mu, Sigma, actual_freq, output_dir)

    # Summary
    print("\nPIPELINE COMPLETED")
    print(f"Training data shape: {train.shape}")
    print(f"Test data shape: {test.shape}")
    print(f"Statistical parameters: mu{mu.shape}, Sigma{Sigma.shape}")
    print(f"Actual crash probability: {actual_freq:.6f}")
    print(f"\nResults saved to: {output_dir.absolute()}")

    # Return all results
    return {
        "mu": mu,
        "Sigma": Sigma,
        "actual_freq": actual_freq
    }


if __name__ == "__main__":
    # Execute pipeline with default parameters
    results = run_pipeline()