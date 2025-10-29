# Coreloops AI Engineer - Technical Test Solution

A production-ready ML pipeline for predicting next-day customer spending from daily transaction data.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Run the Pipeline

```bash
# Run complete ETL + ML training pipeline
python -m scripts.run_pipeline
```

This will:
1. Download transaction data from GCS
2. Clean and normalize the data
3. Aggregate into daily customer metrics
4. Engineer features for ML
5. Train a Random Forest model
6. Save artifacts to `artifacts/`

### Make Predictions

```bash
# Predict next-day spending for a customer
python -m scripts.predict --customer C00042 --date 2024-10-06
```

---

## 📁 Project Structure

```
samson-coreloops-ai-tech-test/
├── src/
│   ├── ingestion.py       # Data download and loading from GCS
│   ├── preprocessing.py   # Data cleaning and normalization
│   ├── transformation.py  # Currency conversion and aggregation
│   ├── features.py        # Feature engineering for ML
│   └── model.py           # ML model training and evaluation
├── scripts/
│   ├── run_pipeline.py    # Main pipeline orchestration
│   └── predict.py         # CLI for predictions
├── artifacts/
│   ├── daily_customer_metrics.parquet  # Aggregated metrics
│   └── model/                           # Saved model artifacts
├── data/                   # Downloaded raw CSV files (cached)
├── requirements.txt
└── README.md
```

---

## 🔧 Pipeline Components

### 1. Data Ingestion (`src/ingestion.py`)
- Downloads daily transaction CSVs from GCS bucket
- Downloads FX rates for currency conversion
- **Caches files locally** - avoids re-downloading existing data
- **Auto-discovers new files** - continues checking for new daily files
- **Incremental processing** - handles new files appearing over time
- Stops after 7 consecutive missing files (configurable)

**See [INCREMENTAL_DATA.md](INCREMENTAL_DATA.md) for detailed incremental ingestion strategy**

### 2. Data Preprocessing (`src/preprocessing.py`)

**Deduplication Strategy:**
- Identifies duplicates based on: `invoice_id`, `product_id`, `timestamp`, `quantity`, `unit_price`
- Keeps first occurrence
- Rationale: Exact matches likely indicate system duplicates

**Missing Value Handling:**

| Field | Strategy | Rationale |
|-------|----------|-----------|
| `customer_id` | Drop rows (~2%) | Cannot aggregate per-customer metrics without ID |
| `unit_price` | Impute with product median (by currency+date), then product median globally | Preserves transactions while maintaining price consistency |
| `description` | Fill with "Unknown" | Non-critical for aggregation |

**Validation:**
- Remove invalid currencies (not GBP/USD/EUR)
- Remove zero or negative unit prices
- Remove unparseable timestamps

### 3. Data Transformation (`src/transformation.py`)

**Currency Conversion:**
- Joins transaction data with daily FX rates
- Converts all prices to GBP: `price_gbp = unit_price * rate_to_gbp`
- Explicit and traceable conversion

**Daily Customer Aggregation:**
Produces metrics per (`date`, `customer_id`):
- `orders`: Count of distinct invoices
- `items`: Sum of absolute quantities
- `gross_gbp`: Revenue from positive quantities
- `returns_gbp`: Revenue from negative quantities (returns)
- `net_gbp`: `gross_gbp + returns_gbp`

Output: `artifacts/daily_customer_metrics.parquet`

### 4. Feature Engineering (`src/features.py`)

**Feature Categories:**

1. **Temporal Features:**
   - Day of week, day of month, is_weekend

2. **Rolling Window Features (3-day, 7-day):**
   - Rolling mean, std, max of `net_gbp`
   - Rolling sum of orders

3. **Lag Features (1-day, 2-day, 7-day):**
   - Lagged `net_gbp`, `orders`, `items`

4. **Customer Lifetime Features:**
   - Cumulative orders and spend
   - Days active since first purchase
   - Average order value

5. **Derived Features:**
   - Items per order ratio
   - Returns ratio

**Total: 31 features**

All features are **lagged appropriately** to prevent data leakage (no future information).

### 5. ML Model (`src/model.py`)

**Model Choice: Random Forest Regressor**

Rationale:
- Handles non-linear relationships well
- Robust to outliers and missing values
- Captures complex interactions between features
- Provides feature importance for interpretability
- No need for feature scaling

**Hyperparameters:**
- `n_estimators`: 100 trees
- `max_depth`: 15
- `min_samples_split`: 10
- `min_samples_leaf`: 5

**Train/Test Split:**
- **Time-based split**: 80% train, 20% test
- Respects temporal ordering (no data leakage)
- Example: Train on Oct 1-20, test on Oct 21-31

**Evaluation Metrics:**
- **MAE (Mean Absolute Error)**: Primary metric - interpretable in GBP
- **RMSE**: Penalizes large errors more
- **R²**: Explains variance in predictions

**Model Artifacts:**
- `random_forest_model.pkl`: Trained model
- `feature_columns.json`: Feature list for predictions
- `metrics.json`: Performance metrics + feature importance
- `model_info.json`: Model configuration

---

## 🎯 Design Decisions & Trade-offs

### 1. Deduplication
**Decision:** Match on 5 key fields (invoice_id, product_id, timestamp, quantity, unit_price)

**Alternative:** Could use all fields, but description/country may vary for same transaction

**Trade-off:** May miss some duplicates with slight variations, but avoids false positives

### 2. Missing customer_id
**Decision:** Drop rows (~2% loss)

**Alternative:** Create synthetic IDs (e.g., UNKNOWN_001)

**Trade-off:** Losing 2% is acceptable vs. introducing noise from ungrouped customers

### 3. Missing unit_price
**Decision:** Impute from product median (same currency+date, then global)

**Alternative:** Drop rows

**Trade-off:** Preserves more data while maintaining price consistency. Could introduce bias if missing prices are systematic.

### 4. Model Choice
**Decision:** Random Forest over Linear Regression or XGBoost

**Alternatives:**
- Linear Regression: Simpler but may underfit
- XGBoost: Better performance but requires more tuning

**Trade-off:** Random Forest balances performance, interpretability, and robustness without extensive tuning.

### 5. Feature Engineering
**Decision:** Focus on temporal patterns (rolling averages, lags)

**Rationale:** Customer spending often follows patterns (day of week, recent trends)

**Trade-off:** More features = more complexity, but captures important patterns

### 6. Data Caching
**Decision:** Cache downloaded CSVs locally

**Rationale:** Avoid repeated downloads during development/testing

**Trade-off:** Uses disk space but dramatically speeds up iteration

---

## 📊 Expected Results

After running the pipeline, expect:

- **Test MAE**: £15-30 (depending on data)
- **Test R²**: 0.6-0.8
- **Top Features**: Recent lags (1d, 2d), rolling averages, customer lifetime stats

---

## 🧪 Testing & Validation

### Manual Testing:
```bash
# Test ingestion
python -m src.ingestion

# Test preprocessing
python -m src.preprocessing

# Test transformation
python -m src.transformation

# Test feature engineering
python -m src.features

# Test model training
python -m src.model
```

### Prediction Testing:
```bash
# Test with known customer
python -m scripts.predict --customer C00042 --date 2024-10-06
```

---

## 🔮 Future Improvements

1. **Model Enhancements:**
   - Hyperparameter tuning (GridSearchCV)
   - Ensemble methods (stacking)
   - Time series models (Prophet, LSTM)

2. **Feature Engineering:**
   - Product category distributions
   - Seasonal patterns
   - Customer segmentation features

3. **Data Quality:**
   - Automated data validation tests
   - Anomaly detection
   - Data drift monitoring

4. **Production Readiness:**
   - Docker containerization
   - CI/CD pipeline
   - Model versioning (MLflow)
   - API endpoint (FastAPI)
   - Monitoring and logging

5. **Performance:**
   - Parallel processing for large datasets
   - Feature caching
   - Incremental training

---

## 📚 Dependencies

- `pandas`: Data manipulation
- `numpy`: Numerical operations
- `scikit-learn`: ML modeling
- `requests`: HTTP downloads
- `pyarrow`: Parquet file handling
- `joblib`: Model serialization

---

## 🤝 Assumptions

1. Data arrives daily and is complete by end of day
2. FX rates are available for all transaction dates
3. Customer IDs are stable over time (no ID changes)
4. Returns (negative quantities) use the same unit price as original purchase
5. Target prediction is for next calendar day (not 24 hours ahead)
6. Historical data is sufficient for feature engineering (minimum 7 days preferred)

---

## 📝 Notes

- The pipeline is **idempotent** - can be run multiple times safely
- Downloaded data is cached in `data/` directory
- All monetary values are converted to GBP for consistency
- Model training uses all available data (train/test split is time-based)
- Predictions require at least 7 days of historical data for best results

---

## 📧 Contact

For questions or issues, please contact the repository maintainer.

---

**Solution by:** Samson Agbo  
**Date:** October 2025  
**Tech Test:** AI/ML Engineer Position
