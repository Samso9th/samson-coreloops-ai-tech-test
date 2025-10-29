"""
Machine Learning model for predicting next-day customer spending.
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import json


MODEL_DIR = Path(__file__).parent.parent / "artifacts" / "model"


def prepare_train_test_split(df: pd.DataFrame, feature_cols: list, 
                             target_col: str = 'net_gbp',
                             test_size: float = 0.2) -> tuple:
    """
    Prepare time-based train/test split.
    
    Args:
        df: Featured DataFrame
        feature_cols: List of feature column names
        target_col: Target variable name
        test_size: Proportion of data for testing (based on time)
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, train_dates, test_dates)
    """
    print("\n📊 Preparing train/test split...")
    
    # Get relevant columns
    df_model = df[feature_cols + [target_col, 'date', 'customer_id']].copy()
    
    # Check for NaN in target
    nan_target = df_model[target_col].isna().sum()
    if nan_target > 0:
        print(f"  Removing {nan_target} rows with NaN target values")
        df_model = df_model.dropna(subset=[target_col])
    
    # Fill NaN in features with 0 (conservative approach for missing lag values)
    nan_features = df_model[feature_cols].isna().sum().sum()
    if nan_features > 0:
        print(f"  Filling {nan_features} NaN feature values with 0")
        df_model[feature_cols] = df_model[feature_cols].fillna(0)
    
    df_clean = df_model
    
    print(f"  Rows with complete features: {len(df_clean):,}")
    
    # Sort by date for time-based split
    df_clean = df_clean.sort_values('date')
    
    # Calculate split point
    split_idx = int(len(df_clean) * (1 - test_size))
    
    train_data = df_clean.iloc[:split_idx]
    test_data = df_clean.iloc[split_idx:]
    
    X_train = train_data[feature_cols]
    y_train = train_data[target_col]
    X_test = test_data[feature_cols]
    y_test = test_data[target_col]
    
    train_dates = (train_data['date'].min(), train_data['date'].max())
    test_dates = (test_data['date'].min(), test_data['date'].max())
    
    print(f"\n  Train set:")
    print(f"    Samples: {len(X_train):,}")
    print(f"    Date range: {train_dates[0]} to {train_dates[1]}")
    print(f"    Target mean: £{y_train.mean():.2f}, std: £{y_train.std():.2f}")
    
    print(f"\n  Test set:")
    print(f"    Samples: {len(X_test):,}")
    print(f"    Date range: {test_dates[0]} to {test_dates[1]}")
    print(f"    Target mean: £{y_test.mean():.2f}, std: £{y_test.std():.2f}")
    
    return X_train, X_test, y_train, y_test, train_dates, test_dates


def train_model(X_train: pd.DataFrame, y_train: pd.Series, 
               n_estimators: int = 100, max_depth: int = 15,
               random_state: int = 42) -> RandomForestRegressor:
    """
    Train Random Forest model.
    
    Args:
        X_train: Training features
        y_train: Training target
        n_estimators: Number of trees
        max_depth: Maximum depth of trees
        random_state: Random seed
    
    Returns:
        Trained model
    """
    print("\n🌲 Training Random Forest model...")
    
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=random_state,
        n_jobs=-1,
        verbose=0
    )
    
    model.fit(X_train, y_train)
    
    print(f"✓ Model trained with {n_estimators} trees, max_depth={max_depth}")
    
    return model


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Evaluate model performance on train and test sets.
    
    Returns:
        Dictionary of metrics
    """
    print("\n📈 Evaluating model performance...")
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Print results
    print(f"\n  Train Metrics:")
    print(f"    MAE:  £{train_mae:.2f}")
    print(f"    RMSE: £{train_rmse:.2f}")
    print(f"    R²:   {train_r2:.4f}")
    
    print(f"\n  Test Metrics:")
    print(f"    MAE:  £{test_mae:.2f}")
    print(f"    RMSE: £{test_rmse:.2f}")
    print(f"    R²:   {test_r2:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n  Top 10 Important Features:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"    {row['feature']}: {row['importance']:.4f}")
    
    metrics = {
        'train_mae': float(train_mae),
        'test_mae': float(test_mae),
        'train_rmse': float(train_rmse),
        'test_rmse': float(test_rmse),
        'train_r2': float(train_r2),
        'test_r2': float(test_r2),
        'feature_importance': feature_importance.to_dict('records')
    }
    
    return metrics


def save_model(model, feature_cols: list, metrics: dict, 
              model_dir: Path = MODEL_DIR) -> None:
    """
    Save trained model and metadata.
    
    Args:
        model: Trained model
        feature_cols: List of feature column names
        metrics: Dictionary of evaluation metrics
        model_dir: Directory to save model artifacts
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = model_dir / "random_forest_model.pkl"
    joblib.dump(model, model_path)
    print(f"\n💾 Saved model to: {model_path}")
    
    # Save feature columns
    feature_path = model_dir / "feature_columns.json"
    with open(feature_path, 'w') as f:
        json.dump(feature_cols, f, indent=2)
    print(f"💾 Saved feature columns to: {feature_path}")
    
    # Save metrics
    metrics_path = model_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"💾 Saved metrics to: {metrics_path}")
    
    # Save model info
    model_info = {
        'model_type': 'RandomForestRegressor',
        'n_features': len(feature_cols),
        'n_estimators': model.n_estimators,
        'max_depth': model.max_depth,
        'test_mae': metrics['test_mae'],
        'test_rmse': metrics['test_rmse'],
        'test_r2': metrics['test_r2']
    }
    
    info_path = model_dir / "model_info.json"
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    print(f"💾 Saved model info to: {info_path}")


def load_model(model_dir: Path = MODEL_DIR):
    """
    Load trained model and metadata.
    
    Returns:
        Tuple of (model, feature_cols, metrics)
    """
    model_path = model_dir / "random_forest_model.pkl"
    feature_path = model_dir / "feature_columns.json"
    metrics_path = model_dir / "metrics.json"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    model = joblib.load(model_path)
    
    with open(feature_path, 'r') as f:
        feature_cols = json.load(f)
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    return model, feature_cols, metrics


def train_pipeline(df: pd.DataFrame, feature_cols: list) -> None:
    """
    Complete ML training pipeline.
    
    Args:
        df: Featured DataFrame
        feature_cols: List of feature column names
    """
    print("\n" + "=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)
    
    # Prepare data
    X_train, X_test, y_train, y_test, train_dates, test_dates = prepare_train_test_split(
        df, feature_cols
    )
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate
    metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
    
    # Save
    save_model(model, feature_cols, metrics)
    
    print("\n" + "=" * 60)
    print("✓ Training pipeline complete!")
    print("=" * 60)


if __name__ == "__main__":
    from src.ingestion import load_all_data
    from src.preprocessing import preprocess_transactions
    from src.transformation import transform_data
    from src.features import engineer_features, get_feature_columns
    
    # Load and process data
    transactions, fx_rates = load_all_data()
    clean_transactions = preprocess_transactions(transactions)
    daily_metrics = transform_data(clean_transactions, fx_rates)
    featured_data = engineer_features(daily_metrics)
    
    # Train model
    feature_cols = get_feature_columns()
    train_pipeline(featured_data, feature_cols)
