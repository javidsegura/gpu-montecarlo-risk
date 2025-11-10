from fetch_split import fetch_and_split
from preprocessing_stats import compute_stats
from evaluation import compute_metrics

def run_pipeline():
    """
    Execute the complete preprocessing and evaluation pipeline.
    """

    # TODO: read from config, x, k, train ratio, indexes dict, start, end
    start="2015-01-01"
    end="2025-01-01"
    x = 0.02
    k = 5
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
    train_ratio = 0.7
    ######################################################################


    # Step 1: Fetch and split data
    train, test = fetch_and_split(indices=INDICES, start=start, end=end, train_ratio=train_ratio)

    # Step 2: Compute statistical parameters
    mu, Sigma = compute_stats(train)

    # Step 3: Compute evaluation metrics
    actual_freq = compute_metrics(test, x=x, k=k)

    # Summary
    print("PIPELINE COMPLETED")
    print(f"Training data shape: {train.shape}")
    print(f"Test data shape: {test.shape}")
    print(f"Statistical parameters: mu{mu.shape}, Sigma{Sigma.shape}")
    print(f"Actual crash probability: {actual_freq:.6f}")

    #TODO: save results in file align with dev team

    # Return all results
    return {
        "train": train,
        "test": test,
        "mu": mu,
        "Sigma": Sigma,
        "actual_freq": actual_freq
    }


if __name__ == "__main__":
    # Execute pipeline with default parameters
    results = run_pipeline()

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