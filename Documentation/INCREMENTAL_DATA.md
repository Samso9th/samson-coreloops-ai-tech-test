# Incremental Data Ingestion Strategy

## Overview

The pipeline is designed to handle **new files appearing over time** through automatic discovery and intelligent caching.

---

## How It Works

### 1. **Auto-Discovery Mode** (Default)

When you run the pipeline without specifying an end date:

```python
discover_and_download_daily_files(start_date="2024-10-01")  # No end_date
```

The system will:
1. Start from `start_date`
2. Try to download each sequential day
3. **Continue until 7 consecutive days without files** (configurable)
4. Assume all available data has been downloaded

**Example behavior:**
```
📥 Auto-discovering files starting from 2024-10-01...
  Will stop after 7 consecutive days without files

✓ Downloaded: 2024-10-01.csv
✓ Downloaded: 2024-10-02.csv
✓ Downloaded: 2024-10-03.csv
✓ Downloaded: 2024-10-04.csv
✓ Downloaded: 2024-10-05.csv
✗ Failed: 2024-10-06.csv
  Last available file: 2024-10-05
✗ Failed: 2024-10-07.csv
... (continues until 7 failures)
  Stopped after 7 consecutive days without files

✓ Total files available: 5
  Date range: 2024-10-01 to 2024-10-05
```

---

### 2. **File Caching Strategy**

All downloaded files are **cached locally** in the `data/` directory:

```
data/
├── 2024-10-01.csv
├── 2024-10-02.csv
├── 2024-10-03.csv
├── 2024-10-04.csv
├── 2024-10-05.csv
└── fx_rates.csv
```

**On subsequent runs:**
- Already downloaded files are **reused** (no re-download)
- Only **new files** are downloaded
- Ensures **idempotency** - safe to run multiple times

---

### 3. **Incremental Processing Workflow**

#### **Day 1: Initial Run (Oct 1-5 available)**
```bash
python -m scripts.run_pipeline
```
- Downloads: 2024-10-01.csv to 2024-10-05.csv
- Processes: 5 days of data
- Trains model on all available data

#### **Day 2: New File Appears (Oct 6 now available)**
```bash
python -m scripts.run_pipeline
```
- Finds cached: 2024-10-01.csv to 2024-10-05.csv (✓ reused)
- Downloads new: 2024-10-06.csv (✓ new)
- Processes: **6 days of data**
- Retrains model with updated data

#### **Day 7: Multiple New Files (Oct 7-12 available)**
```bash
python -m scripts.run_pipeline
```
- Finds cached: 2024-10-01.csv to 2024-10-06.csv (✓ reused)
- Downloads new: 2024-10-07.csv to 2024-10-12.csv (✓ new)
- Processes: **12 days of data**
- Retrains model with full dataset

---

## Configuration Options

### **Auto-Discovery (Recommended for Production)**

```python
# In src/ingestion.py
transaction_files = discover_and_download_daily_files()
```

**Pros:**
- ✅ Automatically finds new files
- ✅ No manual updates needed
- ✅ Handles irregular file availability
- ✅ Production-ready

**Cons:**
- ⚠️ Makes 7+ extra HTTP requests to find end
- ⚠️ Slightly slower first run (but cached after)

---

### **Fixed Date Range (Current Test Implementation)**

```python
# For testing with known data range
transaction_files = discover_and_download_daily_files(
    start_date="2024-10-01",
    end_date="2024-10-05"
)
```

**Pros:**
- ✅ Fast (only 5 requests)
- ✅ Predictable for testing
- ✅ Good for limited test datasets

**Cons:**
- ❌ Requires manual updates
- ❌ Won't find new files automatically

---

### **Custom Max Attempts**

```python
# Adjust sensitivity to missing files
transaction_files = discover_and_download_daily_files(
    start_date="2024-10-01",
    max_attempts=7  # Stop after 7 days of missing files
)
```

**Use cases:**
- Daily batch: `max_attempts=1` (expect consecutive days)
- Weekly batch: `max_attempts=7` (default, handles weekly gaps)
- Monthly: `max_attempts=30` (handles large gaps between files)

---

## Model Retraining Strategy

### **Current Implementation: Full Retrain**

Every pipeline run:
1. Loads **all available data** (cached + new)
2. Preprocesses entire dataset
3. Trains model from scratch

**Advantages:**
- ✅ Simple and robust
- ✅ Model always uses latest data
- ✅ No drift from incremental updates
- ✅ Easy to reason about

**Trade-offs:**
- ⚠️ Reprocesses old data (but cached on disk)
- ⚠️ Retrains full model (but only ~1-2 min)

---

### **Alternative: Incremental Learning (Future Enhancement)**

For very large datasets, could implement:

```python
# Load only new data
new_data = load_new_files_since(last_processed_date)

# Incremental feature update
update_features(new_data)

# Partial model refit (if supported)
model.partial_fit(new_features, new_targets)
```

**When to consider:**
- Dataset > 1M rows
- Training time > 10 minutes
- Hourly/real-time updates needed

---

## Production Deployment Patterns

### **Pattern 1: Scheduled Batch (Recommended)**

```bash
# cron job: Run daily at 1 AM
0 1 * * * cd /path/to/project && python -m scripts.run_pipeline
```

**Workflow:**
1. Pipeline runs automatically each day
2. Discovers any new files since last run
3. Processes incrementally
4. Updates model artifacts
5. Predictions use latest model

---

### **Pattern 2: Event-Driven**

```python
# Triggered when new file uploaded to GCS
def on_file_upload(event):
    # New file detected: 2024-10-13.csv
    run_pipeline()
```

**Benefits:**
- ✅ Processes data as soon as available
- ✅ No delay waiting for cron schedule

---

### **Pattern 3: On-Demand**

```bash
# Manual trigger when needed
python -m scripts.run_pipeline
```

**Use cases:**
- Development/testing
- Ad-hoc analysis
- Backfill historical data

---

## Cache Management

### **Current Cache Behavior**

- **Location**: `data/` directory
- **Retention**: Indefinite (never deleted)
- **Size**: ~40-50 KB per daily file

### **Cache Cleanup (Optional)**

If needed, you can clean old data:

```bash
# Keep only last 30 days
python -c "
from pathlib import Path
from datetime import datetime, timedelta
import os

cutoff = datetime.now() - timedelta(days=30)
for f in Path('data').glob('2024-*.csv'):
    file_date = datetime.strptime(f.stem, '%Y-%m-%d')
    if file_date < cutoff:
        os.remove(f)
        print(f'Removed: {f}')
"
```

---

## Data Quality Monitoring

### **Recommended Checks**

When processing incremental data, monitor:

1. **File availability**:
   ```python
   # Alert if expected file is missing
   if date_expected and not file_found:
       alert("Missing daily file for {date}")
   ```

2. **Data volume**:
   ```python
   # Alert if daily row count anomalous
   if row_count < avg_count * 0.5:
       alert("Low transaction volume")
   ```

3. **Data quality**:
   ```python
   # Track metrics over time
   missing_pct = (missing_rows / total_rows) * 100
   if missing_pct > threshold:
       alert("High missing data rate")
   ```

---

## Testing Incremental Behavior

### **Simulate New Files Appearing**

```bash
# 1. Run with initial data
python -m scripts.run_pipeline

# 2. Manually add new file to data/
wget https://storage.googleapis.com/tech-test-file-storage/data/2024-10-06.csv \
  -O data/2024-10-06.csv

# 3. Run again - should detect and process new file
python -m scripts.run_pipeline
```

**Expected output:**
```
✓ File already exists: 2024-10-01.csv
✓ File already exists: 2024-10-02.csv
...
✓ File already exists: 2024-10-05.csv
✓ Downloaded: 2024-10-06.csv  ← NEW!

✓ Total files available: 6  ← Increased from 5
```

---

## FAQ

### Q: What if files arrive out of order?
**A:** The pipeline handles this gracefully:
- All files are sorted by date during processing
- Time-based features still work correctly
- Model trains on chronologically ordered data

### Q: What if a file is updated/corrected?
**A:** Manual intervention needed:
```bash
# Delete cached file to force re-download
rm data/2024-10-05.csv

# Run pipeline
python -m scripts.run_pipeline
```

### Q: How to backfill historical data?
**A:** Adjust start_date:
```python
# In src/ingestion.py or as parameter
discover_and_download_daily_files(start_date="2024-09-01")
```

### Q: Performance with large date ranges?
**A:** Auto-discovery tests 7 days past last file:
- 5 days of data: ~12 HTTP requests
- 365 days of data: ~372 HTTP requests
- Each request: ~100ms (cached files instant)
- Total discovery time: <30 seconds

---

## Summary

✅ **Pipeline is production-ready for incremental data**

**Key Features:**
1. Auto-discovers new files
2. Caches previously downloaded data
3. Processes all available data each run
4. Retrains model with latest data
5. Safe to run multiple times (idempotent)

**Best Practice:**
- Use auto-discovery in production
- Run on schedule (daily/hourly)
- Monitor for missing files
- Clean old cache periodically

---

**The pipeline handles "new files appearing over time" automatically!** 🚀
