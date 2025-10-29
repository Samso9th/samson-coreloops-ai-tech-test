"""
CLI script for making predictions on customer next-day spending.

Usage:
    python -m scripts.predict --customer C00042 --date 2024-10-06
"""
import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model import load_model


METRICS_PATH = Path(__file__).parent.parent / "artifacts" / "daily_customer_metrics.parquet"


def load_customer_data(customer_id: str, target_date: str) -> pd.DataFrame:
    """
    Load historical data for a customer up to the target date.
    
    Args:
        customer_id: Customer identifier
        target_date: Target prediction date (YYYY-MM-DD)
    
    Returns:
        DataFrame with historical customer metrics
    """
    if not METRICS_PATH.exists():
        raise FileNotFoundError(
            f"Metrics file not found at {METRICS_PATH}. "
            "Please run the pipeline first: python -m scripts.run_pipeline"
        )
    
    # Load all metrics
    df = pd.read_parquet(METRICS_PATH)
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter for customer
    customer_data = df[df['customer_id'] == customer_id].copy()
    
    if len(customer_data) == 0:
        raise ValueError(f"No historical data found for customer {customer_id}")
    
    # Filter up to (but not including) target date
    target_dt = pd.to_datetime(target_date)
    customer_data = customer_data[customer_data['date'] < target_dt]
    
    if len(customer_data) == 0:
        raise ValueError(
            f"No historical data found for customer {customer_id} before {target_date}"
        )
    
    return customer_data


def prepare_prediction_features(customer_data: pd.DataFrame, 
                                target_date: str,
                                feature_cols: list) -> pd.DataFrame:
    """
    Prepare features for prediction by engineering features on historical data.
    
    Args:
        customer_data: Historical customer metrics
        target_date: Target prediction date
        feature_cols: List of required feature columns
    
    Returns:
        DataFrame with single row of features for prediction
    """
    from features import engineer_features
    
    target_dt = pd.to_datetime(target_date)
    
    # Create a dummy row for target date to generate features
    dummy_row = pd.DataFrame([{
        'date': target_dt,
        'customer_id': customer_data['customer_id'].iloc[0],
        'orders': 0,  # Placeholder, not used in prediction
        'items': 0,
        'gross_gbp': 0,
        'returns_gbp': 0,
        'net_gbp': 0  # This is what we're predicting
    }])
    
    # Combine with historical data
    combined = pd.concat([customer_data, dummy_row], ignore_index=True)
    
    # Engineer features
    featured = engineer_features(combined)
    
    # Get the last row (target date)
    prediction_row = featured[featured['date'] == target_dt].copy()
    
    # Check if all required features are present
    missing_features = set(feature_cols) - set(prediction_row.columns)
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    # Check for NaN values
    feature_values = prediction_row[feature_cols]
    if feature_values.isna().any().any():
        nan_features = feature_values.columns[feature_values.isna().any()].tolist()
        print(f"⚠️  Warning: Some features have NaN values: {nan_features}")
        print("   This may occur if there's insufficient historical data.")
        # Fill NaN with 0 (conservative approach)
        feature_values = feature_values.fillna(0)
    
    return feature_values


def make_prediction(customer_id: str, target_date: str) -> dict:
    """
    Make a prediction for a customer on a specific date.
    
    Args:
        customer_id: Customer identifier
        target_date: Target prediction date (YYYY-MM-DD)
    
    Returns:
        Dictionary with prediction results
    """
    # Load model
    try:
        model, feature_cols, metrics = load_model()
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Model not found. Please run the pipeline first: python -m scripts.run_pipeline"
        )
    
    # Load customer data
    customer_data = load_customer_data(customer_id, target_date)
    
    # Prepare features
    X = prepare_prediction_features(customer_data, target_date, feature_cols)
    
    # Make prediction
    prediction = model.predict(X)[0]
    
    # Get historical context
    recent_avg = customer_data.tail(7)['net_gbp'].mean()
    recent_std = customer_data.tail(7)['net_gbp'].std()
    total_days = len(customer_data)
    
    return {
        'customer_id': customer_id,
        'target_date': target_date,
        'predicted_net_gbp': prediction,
        'historical_days': total_days,
        'recent_7d_avg': recent_avg,
        'recent_7d_std': recent_std,
        'model_test_mae': metrics['test_mae']
    }


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Predict next-day customer spending',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m scripts.predict --customer C00042 --date 2024-10-06
  python -m scripts.predict --customer C12345 --date 2024-10-15
        """
    )
    
    parser.add_argument(
        '--customer',
        required=True,
        help='Customer ID (e.g., C00042)'
    )
    
    parser.add_argument(
        '--date',
        required=True,
        help='Target date for prediction (YYYY-MM-DD format)'
    )
    
    args = parser.parse_args()
    
    # Validate date format
    try:
        datetime.strptime(args.date, '%Y-%m-%d')
    except ValueError:
        print(f"❌ Error: Invalid date format '{args.date}'. Use YYYY-MM-DD format.")
        sys.exit(1)
    
    # Make prediction
    try:
        print("\n" + "=" * 60)
        print("CUSTOMER REVENUE PREDICTION")
        print("=" * 60)
        print(f"\n🔮 Predicting for customer {args.customer} on {args.date}...\n")
        
        result = make_prediction(args.customer, args.date)
        
        print("=" * 60)
        print("PREDICTION RESULT")
        print("=" * 60)
        print(f"\n  Customer ID:           {result['customer_id']}")
        print(f"  Target Date:           {result['target_date']}")
        print(f"\n  💰 Predicted Net GBP:  £{result['predicted_net_gbp']:.2f}")
        print(f"\n  Historical Context:")
        print(f"    Days of history:     {result['historical_days']}")
        print(f"    Recent 7-day avg:    £{result['recent_7d_avg']:.2f}")
        print(f"    Recent 7-day std:    £{result['recent_7d_std']:.2f}")
        print(f"\n  Model Performance:")
        print(f"    Test MAE:            £{result['model_test_mae']:.2f}")
        print("\n" + "=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
