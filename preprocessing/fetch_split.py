import numpy as np
import pandas as pd
from pathlib import Path


def fetch_and_split(indices: dict, start="2015-01-01", end="2025-01-01", train_ratio=0.7):
    """
    Load market data from CSV files and split into train/test sets.

    Args:
        start: Start date for filtering (YYYY-MM-DD)
        end: End date for filtering (YYYY-MM-DD)
        indices: Dictionary mapping index names to CSV filenames
        train_ratio: Ratio of data to use for training (default: 0.7)

    Returns:
        train: Training DataFrame with returns (Date index, columns=index names)
        test: Test DataFrame with returns
    """

    print(f"\nLoading index data from CSV files...")
    print(f"  Date range: {start} to {end}")

    dataset_dir = Path(__file__).parent.parent / "dataset"
    
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_dir}. "
            "Please ensure the dataset folder exists with CSV files."
        )

    # Step 1: Read all CSV files
    dataframes = {}
    missing_files = []
    
    for index_name, csv_filename in indices.items():
        csv_path = dataset_dir / csv_filename
        
        if not csv_path.exists():
            missing_files.append(f"{index_name} -> {csv_filename}")
            print(f"  WARNING: CSV file not found: {csv_filename} (skipping {index_name})")
            continue
        
        try:
            # Read CSV file
            df = pd.read_csv(csv_path)
            
            # Verify required columns exist
            if 'Date' not in df.columns or 'Return' not in df.columns:
                print(f"  WARNING: {csv_filename} missing 'Date' or 'Return' column (skipping {index_name})")
                continue
            
            # Parse dates and set as index
            df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y', errors='coerce')
            
            # Drop rows with invalid dates
            df = df.dropna(subset=['Date'])
            
            # Set Date as index
            df = df.set_index('Date')
            
            # Extract Return column and rename to index name
            dataframes[index_name] = df['Return']
            
            print(f"  ✓ Loaded {index_name}: {len(df)} rows from {csv_filename}")
            
        except Exception as e:
            print(f"  ERROR: Failed to load {csv_filename} for {index_name}: {e}")
            continue
    
    if missing_files:
        print(f"\n  Missing files: {len(missing_files)}")
        for msg in missing_files:
            print(f"    - {msg}")
    
    if len(dataframes) == 0:
        raise ValueError(
            "No data was successfully loaded. "
            "Check CSV filenames in config.yaml and ensure dataset folder contains the files."
        )
    
    print(f"\nSuccessfully loaded {len(dataframes)} indices")

    # Step 2: Combine all indices into single DataFrame
    # Inner join to keep only dates present in all indices
    returns = pd.DataFrame(dataframes)
    
    print(f"  Combined data: {len(returns)} rows before alignment")
    
    # Step 3: Filter by date range
    start_date = pd.to_datetime(start)
    end_date = pd.to_datetime(end)
    
    returns = returns.loc[start_date:end_date]
    
    if len(returns) == 0:
        raise ValueError(
            f"No data available in date range {start} to {end}. "
            "Check date range and CSV file date ranges."
        )
    
    # Step 4: Drop rows with any NaN values (missing data for any index)
    returns_before_dropna = len(returns)
    returns = returns.dropna()
    rows_dropped = returns_before_dropna - len(returns)
    
    if rows_dropped > 0:
        print(f"  Dropped {rows_dropped} rows with missing data (after alignment)")
    
    if returns.empty or len(returns) < 100:
        raise ValueError(
            f"Insufficient data: only {len(returns)} days available after processing. "
            f"Minimum required: 100 days"
        )

    # Step 5: Split train/test
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
    # Example indices mapping (update config.yaml with actual CSV filenames)
    INDICES = {
        "AEX": "AEX.csv",
        "Bel20": "Bel20.csv",
        "Stoxx50": "Stoxx50.csv",
        "SMI": "SMI.csv",
        "Ibex35": "Ibex35.csv",
    }
    train, test = fetch_and_split(indices=INDICES, train_ratio=0.7)
    print(f"\nTrain shape: {train.shape}")
    print(f"Test shape: {test.shape}")
