import yfinance as yf
import numpy as np
import pandas as pd


def fetch_and_split(indices:dict, start="2015-01-01", end="2025-01-01", train_ratio=0.7):
    """
    Download market data and split into train/test sets.

    Args:
        start: Start date for data download
        end: End date for data download
        indices: Dictionary mapping index names to ticker symbols
        train_ratio: Ratio of data to use for training (default: 0.7)

    Returns:
        train: Training DataFrame with returns
        test: Test DataFrame with returns
    """

    print(f"\nDownloading index data from Yahoo Finance...")
    print(f"  Date range: {start} to {end}")

    tickers = list(indices.values())

    data = yf.download(tickers, start=start, end=end, auto_adjust=True)["Close"]

    # Clean data
    data = data.dropna(axis=1, how='all')

    if data.empty or len(data.columns) == 0:
        raise ValueError(
            "No data was successfully downloaded. "
            "Check ticker symbols and date range."
        )

    print(f"Successfully downloaded {len(data.columns)} indices")

    # Calculate returns (simple returns, not log returns)
    returns = (data / data.shift(1) - 1).dropna()

    # TODO: research further on the impact of log returns
    # Alternative: returns = np.log(data / data.shift(1)).dropna()

    # Drop any remaining columns with NaN values
    returns = returns.dropna(axis=1)

    if returns.empty or len(returns) < 100:
        raise ValueError(
            f"Insufficient data: only {len(returns)} days available. "
            f"Minimum required: 100 days"
        )

    # Split train/test
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(
            f"Invalid train_ratio={train_ratio}. It must be between 0 and 1 "
            "so both train and test sets are non-empty."
        )
    split_idx = int(len(returns) * train_ratio)
    train = returns.iloc[:split_idx]
    test = returns.iloc[split_idx:]

    print(f"\nData split:")
    print(f"Total: {len(returns)} days, {returns.shape[1]} indices")
    print(f"Train: {len(train)} days ({train.index[0].date()} to {train.index[-1].date()})")
    print(f"Test:  {len(test)} days ({test.index[0].date()} to {test.index[-1].date()})")

    return train, test


if __name__ == "__main__":
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
    print(f"\nTrain shape: {train.shape}")
    print(f"Test shape: {test.shape}")
