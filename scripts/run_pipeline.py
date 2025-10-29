"""
Main pipeline script to run the complete ETL and ML training workflow.

Usage:
    python -m scripts.run_pipeline                    # Auto-discover new files
    python -m scripts.run_pipeline --end 2024-10-05  # Fixed date range
"""
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ingestion import load_all_data
from preprocessing import preprocess_transactions
from transformation import transform_data
from features import engineer_features, get_feature_columns
from model import train_pipeline


def main():
    """Run the complete pipeline."""
    parser = argparse.ArgumentParser(
        description='Run the complete ML pipeline',
        epilog='Example: python -m scripts.run_pipeline --end 2024-10-05'
    )
    parser.add_argument(
        '--end',
        type=str,
        default=None,
        help='End date for data ingestion (YYYY-MM-DD). If not specified, auto-discovers all available files.'
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("CORELOOPS ML PIPELINE - CUSTOMER REVENUE PREDICTION")
    print("=" * 60)
    
    if args.end:
        print(f"Mode: Fixed date range (ending {args.end})")
    else:
        print("Mode: Auto-discovery (will find all available files)")
    
    try:
        # Step 1: Data Ingestion
        transactions, fx_rates = load_all_data(end_date=args.end)
        
        # Step 2: Data Preprocessing
        clean_transactions = preprocess_transactions(transactions)
        
        # Step 3: Data Transformation
        daily_metrics = transform_data(clean_transactions, fx_rates)
        
        # Step 4: Feature Engineering
        featured_data = engineer_features(daily_metrics)
        
        # Step 5: Model Training
        feature_cols = get_feature_columns()
        train_pipeline(featured_data, feature_cols)
        
        print("\n" + "=" * 60)
        print("✅ PIPELINE COMPLETE!")
        print("=" * 60)
        print("\nGenerated artifacts:")
        print("  📊 artifacts/daily_customer_metrics.parquet")
        print("  🤖 artifacts/model/random_forest_model.pkl")
        print("  📋 artifacts/model/feature_columns.json")
        print("  📈 artifacts/model/metrics.json")
        print("\nNext steps:")
        print("  Run predictions with:")
        print("    python -m scripts.predict --customer C00042 --date 2024-10-06")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
