"""
Data preprocessing module for cleaning and normalizing transaction data.
"""
import pandas as pd
import numpy as np


def deduplicate_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate transaction rows.
    
    Deduplication criteria: Same invoice_id, product_id, timestamp, quantity, unit_price
    Keep first occurrence.
    """
    print("\n🔍 Deduplicating transactions...")
    initial_count = len(df)
    
    # Define columns to check for duplicates
    duplicate_cols = ['invoice_id', 'product_id', 'timestamp', 'quantity', 'unit_price']
    
    # Keep first occurrence
    df_clean = df.drop_duplicates(subset=duplicate_cols, keep='first')
    
    duplicates_removed = initial_count - len(df_clean)
    print(f"✓ Removed {duplicates_removed} duplicate rows ({duplicates_removed/initial_count*100:.2f}%)")
    
    return df_clean


def handle_missing_customer_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing customer_id values.
    
    Strategy: Drop rows with missing customer_id (typically ~2% of data).
    Alternative could be to create synthetic IDs, but this may skew per-customer predictions.
    """
    print("\n🔍 Handling missing customer_id...")
    initial_count = len(df)
    
    missing_count = df['customer_id'].isna().sum()
    print(f"  Missing customer_id: {missing_count} rows ({missing_count/initial_count*100:.2f}%)")
    
    # Drop rows with missing customer_id
    df_clean = df.dropna(subset=['customer_id']).copy()
    
    removed = initial_count - len(df_clean)
    print(f"✓ Dropped {removed} rows with missing customer_id")
    
    return df_clean


def handle_missing_unit_price(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing unit_price values.
    
    Strategy: Impute using median price per product+currency+day.
    If no median available for that combination, use global product median.
    Drop rows where imputation is not possible.
    """
    print("\n🔍 Handling missing unit_price...")
    initial_missing = df['unit_price'].isna().sum()
    print(f"  Missing unit_price: {initial_missing} rows ({initial_missing/len(df)*100:.2f}%)")
    
    if initial_missing == 0:
        print("✓ No missing unit_price values")
        return df
    
    df_clean = df.copy()
    
    # Calculate median price per product, currency, and date
    df_clean['price_key'] = df_clean['product_id'] + '_' + df_clean['currency'] + '_' + df_clean['file_date']
    
    # First try: impute with product+currency+date median
    for idx in df_clean[df_clean['unit_price'].isna()].index:
        price_key = df_clean.loc[idx, 'price_key']
        product_id = df_clean.loc[idx, 'product_id']
        
        # Get median for same product+currency+date
        same_key_prices = df_clean[(df_clean['price_key'] == price_key) & 
                                     (df_clean['unit_price'].notna())]['unit_price']
        
        if len(same_key_prices) > 0:
            df_clean.loc[idx, 'unit_price'] = same_key_prices.median()
        else:
            # Fallback: use product median across all dates/currencies
            product_prices = df_clean[(df_clean['product_id'] == product_id) & 
                                       (df_clean['unit_price'].notna())]['unit_price']
            if len(product_prices) > 0:
                df_clean.loc[idx, 'unit_price'] = product_prices.median()
    
    df_clean.drop(columns=['price_key'], inplace=True)
    
    # Drop any remaining rows with missing unit_price
    remaining_missing = df_clean['unit_price'].isna().sum()
    if remaining_missing > 0:
        df_clean = df_clean.dropna(subset=['unit_price'])
        print(f"  Could not impute {remaining_missing} rows, dropped them")
    
    imputed = initial_missing - remaining_missing
    print(f"✓ Imputed {imputed} missing unit_price values, dropped {remaining_missing}")
    
    return df_clean


def handle_missing_description(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing description values.
    
    Strategy: Fill with "Unknown" as descriptions are not critical for aggregation.
    """
    print("\n🔍 Handling missing descriptions...")
    missing_count = df['description'].isna().sum()
    
    if missing_count > 0:
        df['description'] = df['description'].fillna("Unknown")
        print(f"✓ Filled {missing_count} missing descriptions with 'Unknown'")
    else:
        print("✓ No missing descriptions")
    
    return df


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate data types and ranges.
    
    - Ensure currencies are valid (GBP, USD, EUR)
    - Ensure timestamps are parseable
    - Remove rows with zero unit_price
    """
    print("\n🔍 Validating data...")
    initial_count = len(df)
    
    # Validate currencies
    valid_currencies = ['GBP', 'USD', 'EUR']
    invalid_currency = ~df['currency'].isin(valid_currencies)
    if invalid_currency.sum() > 0:
        print(f"  Removing {invalid_currency.sum()} rows with invalid currency")
        df = df[~invalid_currency].copy()
    
    # Remove rows with zero or negative unit_price
    invalid_price = df['unit_price'] <= 0
    if invalid_price.sum() > 0:
        print(f"  Removing {invalid_price.sum()} rows with invalid unit_price (<=0)")
        df = df[~invalid_price].copy()
    
    # Parse timestamps
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    invalid_timestamp = df['timestamp'].isna()
    if invalid_timestamp.sum() > 0:
        print(f"  Removing {invalid_timestamp.sum()} rows with invalid timestamp")
        df = df[~invalid_timestamp].copy()
    
    removed = initial_count - len(df)
    print(f"✓ Validation complete, removed {removed} invalid rows")
    
    return df


def preprocess_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main preprocessing pipeline.
    
    Args:
        df: Raw transaction DataFrame
    
    Returns:
        Cleaned and validated DataFrame
    """
    print("\n" + "=" * 60)
    print("DATA PREPROCESSING")
    print("=" * 60)
    
    initial_count = len(df)
    print(f"Initial rows: {initial_count:,}")
    
    # Apply preprocessing steps
    df = deduplicate_transactions(df)
    df = handle_missing_customer_id(df)
    df = handle_missing_unit_price(df)
    df = handle_missing_description(df)
    df = validate_data(df)
    
    final_count = len(df)
    removed_pct = (initial_count - final_count) / initial_count * 100
    
    print(f"\n✓ Final rows: {final_count:,} (removed {removed_pct:.2f}%)")
    print("=" * 60)
    
    return df


if __name__ == "__main__":
    from src.ingestion import load_all_data
    
    transactions, fx_rates = load_all_data()
    clean_transactions = preprocess_transactions(transactions)
    
    print(f"\nCleaned transactions shape: {clean_transactions.shape}")
    print(f"Columns: {list(clean_transactions.columns)}")
