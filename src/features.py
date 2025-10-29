"""
Feature engineering module for ML model.
"""
import pandas as pd
import numpy as np
from datetime import datetime


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create temporal features from date column.
    
    Features:
    - day_of_week (0=Monday, 6=Sunday)
    - day_of_month
    - is_weekend
    """
    df = df.copy()
    
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_month'] = df['date'].dt.day
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    return df


def create_rolling_features(df: pd.DataFrame, windows: list = [3]) -> pd.DataFrame:
    """
    Create rolling/lagged features for time series prediction.
    
    Features for each window:
    - rolling_mean_net_gbp: average spending
    - rolling_std_net_gbp: spending volatility
    - rolling_max_net_gbp: peak spending
    - rolling_sum_orders: total orders
    
    Args:
        df: DataFrame with date and customer_id
        windows: List of window sizes in days (reduced to [3] for small datasets)
    """
    df = df.copy()
    df = df.sort_values(['customer_id', 'date'])
    
    for window in windows:
        # Rolling aggregations per customer
        df[f'rolling_{window}d_mean_net'] = df.groupby('customer_id')['net_gbp'].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        )
        
        df[f'rolling_{window}d_std_net'] = df.groupby('customer_id')['net_gbp'].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).std()
        )
        
        df[f'rolling_{window}d_max_net'] = df.groupby('customer_id')['net_gbp'].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).max()
        )
        
        df[f'rolling_{window}d_sum_orders'] = df.groupby('customer_id')['orders'].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).sum()
        )
    
    # Fill NaN values
    for window in windows:
        df[f'rolling_{window}d_mean_net'] = df[f'rolling_{window}d_mean_net'].fillna(0)
        df[f'rolling_{window}d_std_net'] = df[f'rolling_{window}d_std_net'].fillna(0)
        df[f'rolling_{window}d_max_net'] = df[f'rolling_{window}d_max_net'].fillna(0)
        df[f'rolling_{window}d_sum_orders'] = df[f'rolling_{window}d_sum_orders'].fillna(0)
    
    return df


def create_lag_features(df: pd.DataFrame, lags: list = [1, 2]) -> pd.DataFrame:
    """
    Create lagged features (previous day values).
    
    Args:
        df: DataFrame sorted by customer_id and date
        lags: List of lag periods in days (reduced to [1, 2] for small datasets)
    """
    df = df.copy()
    df = df.sort_values(['customer_id', 'date'])
    
    for lag in lags:
        df[f'lag_{lag}d_net_gbp'] = df.groupby('customer_id')['net_gbp'].shift(lag)
        df[f'lag_{lag}d_orders'] = df.groupby('customer_id')['orders'].shift(lag)
        df[f'lag_{lag}d_items'] = df.groupby('customer_id')['items'].shift(lag)
    
    # Fill NaN with 0 for customers with insufficient history
    for lag in lags:
        df[f'lag_{lag}d_net_gbp'] = df[f'lag_{lag}d_net_gbp'].fillna(0)
        df[f'lag_{lag}d_orders'] = df[f'lag_{lag}d_orders'].fillna(0)
        df[f'lag_{lag}d_items'] = df[f'lag_{lag}d_items'].fillna(0)
    
    return df


def create_customer_lifetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create customer lifetime statistics (up to current date).
    
    Features:
    - customer_total_orders: cumulative orders
    - customer_total_spend: cumulative spending
    - customer_days_active: days since first purchase
    - customer_avg_order_value: average spending per order
    """
    df = df.copy()
    df = df.sort_values(['customer_id', 'date'])
    
    # Cumulative metrics (excluding current day)
    df['customer_total_orders'] = df.groupby('customer_id')['orders'].cumsum().shift(1)
    df['customer_total_spend'] = df.groupby('customer_id')['net_gbp'].cumsum().shift(1)
    
    # Days since first purchase
    df['first_purchase_date'] = df.groupby('customer_id')['date'].transform('first')
    df['customer_days_active'] = (df['date'] - df['first_purchase_date']).dt.days
    
    # Average order value
    df['customer_avg_order_value'] = df['customer_total_spend'] / df['customer_total_orders'].clip(lower=1)
    
    # Fill NaN for first occurrences
    df['customer_total_orders'] = df['customer_total_orders'].fillna(0)
    df['customer_total_spend'] = df['customer_total_spend'].fillna(0)
    df['customer_avg_order_value'] = df['customer_avg_order_value'].fillna(0)
    
    df.drop(columns=['first_purchase_date'], inplace=True)
    
    return df


def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features from existing metrics.
    
    Features:
    - avg_items_per_order: items / orders
    - returns_ratio: returns_gbp / gross_gbp
    """
    df = df.copy()
    
    # Average items per order
    df['avg_items_per_order'] = df['items'] / df['orders'].clip(lower=1)
    
    # Returns ratio (handle division by zero)
    df['returns_ratio'] = (df['returns_gbp'] / df['gross_gbp'].clip(lower=0.01)).fillna(0)
    df['returns_ratio'] = df['returns_ratio'].clip(upper=0)  # Should be negative or zero
    
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main feature engineering pipeline.
    
    Args:
        df: Daily customer metrics DataFrame
    
    Returns:
        DataFrame with engineered features
    """
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)
    
    initial_cols = len(df.columns)
    print(f"Initial features: {initial_cols}")
    
    # Sort by customer and date
    df = df.sort_values(['customer_id', 'date']).reset_index(drop=True)
    
    # Create features
    print("\n🔧 Creating temporal features...")
    df = create_temporal_features(df)
    
    print("🔧 Creating rolling features...")
    df = create_rolling_features(df, windows=[3])
    
    print("🔧 Creating lag features...")
    df = create_lag_features(df, lags=[1, 2])
    
    print("🔧 Creating customer lifetime features...")
    df = create_customer_lifetime_features(df)
    
    print("🔧 Creating derived features...")
    df = create_derived_features(df)
    
    final_cols = len(df.columns)
    new_features = final_cols - initial_cols
    
    print(f"\n✓ Created {new_features} new features (total: {final_cols})")
    print(f"✓ Feature engineering complete")
    print("=" * 60)
    
    return df


def get_feature_columns() -> list:
    """
    Return list of feature column names for ML model.
    
    Excludes: date, customer_id, target variables (net_gbp, gross_gbp, returns_gbp)
    """
    base_features = ['orders', 'items']
    
    temporal_features = ['day_of_week', 'day_of_month', 'is_weekend']
    
    rolling_features = []
    for window in [3]:  # Reduced for small datasets
        rolling_features.extend([
            f'rolling_{window}d_mean_net',
            f'rolling_{window}d_std_net',
            f'rolling_{window}d_max_net',
            f'rolling_{window}d_sum_orders'
        ])
    
    lag_features = []
    for lag in [1, 2]:  # Reduced for small datasets
        lag_features.extend([
            f'lag_{lag}d_net_gbp',
            f'lag_{lag}d_orders',
            f'lag_{lag}d_items'
        ])
    
    lifetime_features = [
        'customer_total_orders',
        'customer_total_spend',
        'customer_days_active',
        'customer_avg_order_value'
    ]
    
    derived_features = [
        'avg_items_per_order',
        'returns_ratio'
    ]
    
    all_features = (base_features + temporal_features + rolling_features + 
                   lag_features + lifetime_features + derived_features)
    
    return all_features


if __name__ == "__main__":
    from src.ingestion import load_all_data
    from src.preprocessing import preprocess_transactions
    from src.transformation import transform_data
    
    transactions, fx_rates = load_all_data()
    clean_transactions = preprocess_transactions(transactions)
    daily_metrics = transform_data(clean_transactions, fx_rates)
    
    featured_data = engineer_features(daily_metrics)
    
    print(f"\nFeatured data shape: {featured_data.shape}")
    print(f"\nFeature columns:")
    for col in get_feature_columns():
        print(f"  - {col}")
