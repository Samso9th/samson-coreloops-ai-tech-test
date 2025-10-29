"""
Data transformation module for currency conversion and aggregation.
"""
import pandas as pd
import numpy as np
from pathlib import Path


def convert_to_gbp(df: pd.DataFrame, fx_rates: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all monetary values to GBP using daily FX rates.
    
    Args:
        df: Cleaned transaction DataFrame
        fx_rates: FX rates DataFrame with columns [date, currency, rate_to_gbp]
    
    Returns:
        DataFrame with additional 'price_gbp' column
    """
    print("\n💱 Converting currencies to GBP...")
    
    # Ensure date columns are datetime
    df['date'] = pd.to_datetime(df['file_date'])
    fx_rates['date'] = pd.to_datetime(fx_rates['date'])
    
    # Merge with FX rates
    df_with_fx = df.merge(
        fx_rates[['date', 'currency', 'rate_to_gbp']],
        on=['date', 'currency'],
        how='left'
    )
    
    # Check for missing FX rates
    missing_fx = df_with_fx['rate_to_gbp'].isna().sum()
    if missing_fx > 0:
        print(f"⚠️  Warning: {missing_fx} rows missing FX rates")
        # For GBP, rate should be 1.0
        df_with_fx.loc[(df_with_fx['currency'] == 'GBP') & 
                       (df_with_fx['rate_to_gbp'].isna()), 'rate_to_gbp'] = 1.0
    
    # Calculate price in GBP
    df_with_fx['price_gbp'] = df_with_fx['unit_price'] * df_with_fx['rate_to_gbp']
    
    print(f"✓ Converted {len(df_with_fx)} rows to GBP")
    print(f"  Currency distribution:")
    for currency in df_with_fx['currency'].value_counts().items():
        print(f"    {currency[0]}: {currency[1]:,} rows")
    
    return df_with_fx


def aggregate_daily_customer_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate transactions into daily per-customer metrics.
    
    Metrics:
    - date: transaction date
    - customer_id: customer identifier
    - orders: count of distinct invoices
    - items: sum of absolute quantities
    - gross_gbp: sum of (quantity * price_gbp) for positive quantities
    - returns_gbp: sum of (quantity * price_gbp) for negative quantities
    - net_gbp: gross_gbp + returns_gbp
    
    Args:
        df: Transaction DataFrame with price_gbp column
    
    Returns:
        Aggregated DataFrame
    """
    print("\n📊 Aggregating daily customer metrics...")
    
    # Calculate total value in GBP for each row
    df['value_gbp'] = df['quantity'] * df['price_gbp']
    
    # Separate positive (gross) and negative (returns) quantities
    df['gross_value'] = df['value_gbp'].where(df['quantity'] > 0, 0)
    df['returns_value'] = df['value_gbp'].where(df['quantity'] < 0, 0)
    
    # Group by date and customer
    agg_metrics = df.groupby(['date', 'customer_id']).agg({
        'invoice_id': 'nunique',           # distinct invoices
        'quantity': lambda x: abs(x).sum(), # sum of absolute quantities
        'gross_value': 'sum',               # gross revenue
        'returns_value': 'sum',             # returns (negative)
        'value_gbp': 'sum'                  # net revenue
    }).reset_index()
    
    # Rename columns
    agg_metrics.columns = ['date', 'customer_id', 'orders', 'items', 
                           'gross_gbp', 'returns_gbp', 'net_gbp']
    
    # Ensure proper data types
    agg_metrics['orders'] = agg_metrics['orders'].astype(int)
    agg_metrics['items'] = agg_metrics['items'].astype(int)
    
    print(f"✓ Aggregated to {len(agg_metrics)} daily customer records")
    print(f"  Date range: {agg_metrics['date'].min()} to {agg_metrics['date'].max()}")
    print(f"  Unique customers: {agg_metrics['customer_id'].nunique()}")
    print(f"  Total net revenue: £{agg_metrics['net_gbp'].sum():,.2f}")
    
    # Show summary statistics
    print(f"\n  Summary statistics:")
    print(f"    Avg daily spending per customer: £{agg_metrics['net_gbp'].mean():.2f}")
    print(f"    Median daily spending: £{agg_metrics['net_gbp'].median():.2f}")
    print(f"    Avg orders per customer per day: {agg_metrics['orders'].mean():.2f}")
    print(f"    Avg items per customer per day: {agg_metrics['items'].mean():.2f}")
    
    return agg_metrics


def save_metrics(df: pd.DataFrame, output_path: str = None) -> None:
    """
    Save aggregated metrics to Parquet file.
    
    Args:
        df: Aggregated metrics DataFrame
        output_path: Path to save file (defaults to artifacts/daily_customer_metrics.parquet)
    """
    if output_path is None:
        output_path = Path(__file__).parent.parent / "artifacts" / "daily_customer_metrics.parquet"
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_parquet(output_path, index=False, engine='pyarrow')
    print(f"\n💾 Saved metrics to: {output_path}")
    print(f"   File size: {output_path.stat().st_size / 1024:.2f} KB")


def transform_data(transactions: pd.DataFrame, fx_rates: pd.DataFrame) -> pd.DataFrame:
    """
    Main transformation pipeline.
    
    Args:
        transactions: Cleaned transaction DataFrame
        fx_rates: FX rates DataFrame
    
    Returns:
        Aggregated daily customer metrics DataFrame
    """
    print("\n" + "=" * 60)
    print("DATA TRANSFORMATION")
    print("=" * 60)
    
    # Convert to GBP
    transactions_gbp = convert_to_gbp(transactions, fx_rates)
    
    # Aggregate metrics
    daily_metrics = aggregate_daily_customer_metrics(transactions_gbp)
    
    # Save to file
    save_metrics(daily_metrics)
    
    print("=" * 60)
    
    return daily_metrics


if __name__ == "__main__":
    from ingestion import load_all_data
    from preprocessing import preprocess_transactions
    
    transactions, fx_rates = load_all_data()
    clean_transactions = preprocess_transactions(transactions)
    daily_metrics = transform_data(clean_transactions, fx_rates)
    
    print(f"\nFinal metrics shape: {daily_metrics.shape}")
    print(daily_metrics.head(10))
