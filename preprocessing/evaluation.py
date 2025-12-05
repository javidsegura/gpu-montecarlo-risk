import numpy as np
import pandas as pd


def compute_metrics(test, x=0.02, k=5):
    """
    Compute actual crash probability from test data.

    Args:
        test: Test DataFrame with returns (rows=days, columns=indices)
        x: Threshold for crash detection (default: 0.02 = 2% loss)
        k: Minimum number of indices that must crash simultaneously in a single day (default: 5)

    Returns:
        actual_freq: Actual crash probability from test data (frequency of days with >= k crashes)
    """

    # Count how many indices crashed on each day (return < -x)
    actual_crash_days = (test < -x).sum(axis=1)

    # Compute frequency of days where at least k indices crashed simultaneously
    actual_freq = np.mean(actual_crash_days >= k)

    print(f"\nActual crash probability (from test data): {actual_freq:.6f}")
    print(f"Days analyzed: {len(test)}")
    print(f"Days with >= {k} crashes: {np.sum(actual_crash_days >= k)}")

    return actual_freq


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

    actual_freq = compute_metrics(test, x=0.02, k=5)

    print(f"FINAL RESULT: Actual crash frequency = {actual_freq:.6f}")
