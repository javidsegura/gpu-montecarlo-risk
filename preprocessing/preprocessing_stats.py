import numpy as np
import pandas as pd


def compute_stats(train):
    """
    Compute statistical parameters from training data.

    Args:
        train: Training DataFrame with returns

    Returns:
        mu: Mean return vector (numpy array)
        Sigma: Covariance matrix (numpy array)
    """

    # Compute mean returns
    mu = train.mean().values

    # TODO: research further on covariance matrix
    Sigma = train.cov().values

    print(f"\nStatistical parameters computed:")
    print(f"Mean vector (mu): shape {mu.shape}")
    print(f"Covariance matrix (Sigma): shape {Sigma.shape}")
    print(f"Number of indices: {len(mu)}")

    return mu, Sigma


if __name__ == "__main__":
    from fetch_split import fetch_and_split

    INDICES = {
        "STOXX50E": "^STOXX50E",
        "CAC40": "^FCHI",
        "DAX": "^GDAXI",
        "FTSE100": "^FTSE",
        "FTSEMIB": "FTSEMIB.MI",
        "IBEX35": "^IBEX",
        "SMI": "^SSMI",
        "AEX": "^AEX",
        "BEL20": "^BFX"
    }
    train, test = fetch_and_split(indices=INDICES, train_ratio=0.7)
    mu, Sigma = compute_stats(train)

    print(f"\nMean returns (first 5): {mu[:5]}")
    print(f"Covariance matrix (5x5 sample):\n{Sigma[:5, :5]}")