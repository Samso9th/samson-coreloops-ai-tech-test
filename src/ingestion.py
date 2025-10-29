"""
Data ingestion module for downloading and loading transaction data from GCS.
"""
import os
import requests
import pandas as pd
from pathlib import Path
from typing import List, Tuple
from datetime import datetime, timedelta


BASE_URL = "https://storage.googleapis.com/tech-test-file-storage"
DATA_DIR = Path(__file__).parent.parent / "data"


def download_file(url: str, local_path: Path) -> bool:
    """Download a file from URL if it doesn't exist locally."""
    if local_path.exists():
        print(f"✓ File already exists: {local_path.name}")
        return True
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        local_path.parent.mkdir(parents=True, exist_ok=True)
        with open(local_path, 'wb') as f:
            f.write(response.content)
        
        print(f"✓ Downloaded: {local_path.name}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"✗ Failed to download {url}: {e}")
        return False


def download_fx_rates() -> pd.DataFrame:
    """Download and load FX rates data."""
    url = f"{BASE_URL}/fx_rates.csv"
    local_path = DATA_DIR / "fx_rates.csv"
    
    if download_file(url, local_path):
        df = pd.read_csv(local_path)
        df['date'] = pd.to_datetime(df['date'])
        print(f"✓ Loaded FX rates: {len(df)} rows")
        return df
    else:
        raise FileNotFoundError("Failed to download fx_rates.csv")


def discover_and_download_daily_files(start_date: str = "2024-10-01", 
                                       end_date: str = "2024-10-05") -> List[Path]:
    """
    Discover and download daily transaction files from GCS.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format (defaults to 2024-10-05)
    
    Returns:
        List of paths to successfully downloaded files
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    downloaded_files = []
    current_date = start
    
    print(f"\n📥 Discovering files from {start_date} to {end_date}...")
    
    while current_date <= end:
        date_str = current_date.strftime("%Y-%m-%d")
        url = f"{BASE_URL}/data/{date_str}.csv"
        local_path = DATA_DIR / f"{date_str}.csv"
        
        if download_file(url, local_path):
            downloaded_files.append(local_path)
        else:
            # Stop trying if file doesn't exist (bucket may have limited data)
            print(f"  No more files found after {(current_date - timedelta(days=1)).strftime('%Y-%m-%d')}")
            break
        
        current_date += timedelta(days=1)
    
    print(f"\n✓ Total files available: {len(downloaded_files)}")
    
    if len(downloaded_files) == 0:
        raise ValueError("No transaction files found. Check GCS bucket availability.")
    
    return downloaded_files


def load_transaction_files(file_paths: List[Path]) -> pd.DataFrame:
    """Load and combine multiple transaction CSV files."""
    dfs = []
    
    for file_path in file_paths:
        try:
            df = pd.read_csv(file_path)
            # Extract date from filename
            date_str = file_path.stem  # e.g., "2024-10-01"
            df['file_date'] = date_str
            dfs.append(df)
        except Exception as e:
            print(f"✗ Error loading {file_path}: {e}")
    
    if not dfs:
        raise ValueError("No transaction files loaded successfully")
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"✓ Loaded {len(combined)} total transaction rows from {len(dfs)} files")
    
    return combined


def load_all_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main entry point: Download and load all required data.
    
    Returns:
        Tuple of (transactions_df, fx_rates_df)
    """
    print("=" * 60)
    print("DATA INGESTION")
    print("=" * 60)
    
    # Download FX rates
    fx_rates = download_fx_rates()
    
    # Download daily transaction files
    transaction_files = discover_and_download_daily_files()
    
    # Load all transaction data
    transactions = load_transaction_files(transaction_files)
    
    print("\n" + "=" * 60)
    return transactions, fx_rates


if __name__ == "__main__":
    transactions, fx_rates = load_all_data()
    print(f"\nTransactions shape: {transactions.shape}")
    print(f"FX Rates shape: {fx_rates.shape}")
    print(f"\nTransaction columns: {list(transactions.columns)}")
