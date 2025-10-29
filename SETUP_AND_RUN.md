# Setup and Run Instructions

## Quick Setup (5 minutes)

### 1. Install Dependencies

```bash
pip install pandas numpy scikit-learn requests pyarrow joblib
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```

### 2. Run the Complete Pipeline

```bash
python -m scripts.run_pipeline
```

**This will:**
- ✅ Download all transaction data from GCS (Oct 1-5, 2024 - 5 days)
- ✅ Download FX rates
- ✅ Clean and preprocess data
- ✅ Convert all currencies to GBP
- ✅ Aggregate daily customer metrics
- ✅ Engineer features (optimized for small dataset)
- ✅ Train Random Forest model
- ✅ Save artifacts to `artifacts/`

**Expected runtime:** 1-2 minutes

**Note:** The GCS bucket contains 5 days of data (Oct 1-5, 2024). The pipeline has been optimized to work with this limited dataset.

### 3. Make a Prediction

```bash
python -m scripts.predict --customer C00042 --date 2024-10-06
```

**Output example:**
```
============================================================
CUSTOMER REVENUE PREDICTION
============================================================

🔮 Predicting for customer C00042 on 2024-10-06...

============================================================
PREDICTION RESULT
============================================================

  Customer ID:           C00042
  Target Date:           2024-10-06

  💰 Predicted Net GBP:  £45.23

  Historical Context:
    Days of history:     5
    Recent 7-day avg:    £42.15
    Recent 7-day std:    £8.34

  Model Performance:
    Test MAE:            £18.67

============================================================
```

---

## Testing Individual Components

Each module can be run standalone for testing:

### Test Data Ingestion
```bash
python -m src.ingestion
```
Expected: Downloads data and shows file counts

### Test Preprocessing
```bash
python -m src.preprocessing
```
Expected: Shows cleaning steps and removed row counts

### Test Transformation
```bash
python -m src.transformation
```
Expected: Shows currency conversion and aggregation metrics

### Test Feature Engineering
```bash
python -m src.features
```
Expected: Shows 31 features created

### Test Model Training
```bash
python -m src.model
```
Expected: Shows train/test metrics (MAE, RMSE, R²)

---

## Artifacts Generated

After running the pipeline, you'll find:

```
artifacts/
├── daily_customer_metrics.parquet    # Aggregated daily metrics
└── model/
    ├── random_forest_model.pkl       # Trained model
    ├── feature_columns.json          # Feature list
    ├── metrics.json                  # Performance metrics
    └── model_info.json               # Model configuration
```

---

## Troubleshooting

### Issue: "Module not found" errors
**Solution:** Make sure you're running from the project root directory

### Issue: "No module named 'pandas'"
**Solution:** Install dependencies: `pip install -r requirements.txt`

### Issue: "No historical data found for customer"
**Solution:** Run the pipeline first to generate metrics: `python -m scripts.run_pipeline`

### Issue: Network timeout during data download
**Solution:** Check internet connection, script will cache downloaded files and resume

### Issue: "Metrics file not found"
**Solution:** Run the pipeline first: `python -m scripts.run_pipeline`

---

## Data Flow

```
1. GCS Bucket (HTTPS)
   ↓ [download]
2. Raw CSVs (data/)
   ↓ [load]
3. Raw DataFrame
   ↓ [clean, deduplicate, handle nulls]
4. Clean DataFrame
   ↓ [convert currency, aggregate]
5. Daily Customer Metrics (artifacts/*.parquet)
   ↓ [engineer features]
6. Featured Dataset
   ↓ [train/test split]
7. Trained Model (artifacts/model/)
   ↓ [predict]
8. Customer Predictions
```

---

## Expected Performance

| Metric | Expected Value |
|--------|----------------|
| Test MAE | £15-30 |
| Test RMSE | £25-45 |
| Test R² | 0.6-0.8 |
| Training Time | 1-3 minutes |
| Prediction Time | <1 second |

---

## Code Quality Checks

All modules pass Python syntax validation:
```bash
python -m py_compile src/*.py scripts/*.py
```

---

## Next Steps After Setup

1. ✅ Verify all artifacts are generated
2. ✅ Check model metrics in `artifacts/model/metrics.json`
3. ✅ Try predictions with different customers and dates
4. ✅ Review feature importance in metrics.json
5. ✅ Read SOLUTION.md for detailed design decisions

---

## Development Notes

- **Idempotent:** Pipeline can be run multiple times safely
- **Cached:** Downloaded files are cached in `data/` directory
- **Reproducible:** Fixed random seed (42) for model training
- **Modular:** Each component can be tested independently
- **Documented:** Inline comments explain key decisions

---

## Contact

For technical questions about the implementation, refer to:
- **SOLUTION.md** - Comprehensive solution documentation
- **schema.md** - Data schema reference
- **README.md** - Original test requirements

---

**Ready to go!** 🚀

Just run: `python -m scripts.run_pipeline`
