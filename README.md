# Crop Yield Forecast

Machine learning pipeline for predicting crop yields across Turkey using WOFOST crop simulation outputs and real weather data (OpenMeteo).

## Dataset

- **Source:** WOFOST crop simulator + OpenMeteo hourly weather
- **Size:** ~466M rows, 5 GB parquet
- **Coverage:** Turkey grid (lat 36–43, lon 26–44), 2014–2024
- **Crops:** 22 crops (wheat, barley, maize, sugarbeet, etc.)
- **Scenarios:** 3 soil water scenarios — dry / normal / wet

| Column group | Key columns |
|---|---|
| Identity | `latitude`, `longitude`, `crop_name`, `year`, `wav_scenario` |
| Weather | `AIR_TEMP`, `PRECIP`, `AIR_HUMIDITY` |
| WOFOST | `DVS`, `LAI`, `TAGP`, `RFTRA` |
| Target | `harvest_twso` (kg/ha), `sim_success` |

## Project Structure

```
crop-yield-forecast/
│
├── data/
│   ├── raw/                          # Original 5 GB parquet (not tracked by git)
│   └── processed/
│       ├── train.parquet             # Season-level features, years 2014–2022
│       ├── test.parquet              # Season-level features, years 2023–2024
│       └── crop_te_map.json          # Crop target encoding mapping
│
├── notebooks/
│   ├── 01_eda.ipynb                  # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb  # Season aggregation + feature prep
│   ├── 03_baseline_model.ipynb       # Full-season LightGBM (R²=0.964)
│   ├── 04_early_prediction.ipynb     # Early-season forecasting (day 30/60/90/120)
│   └── 05_crop_recommendation.ipynb  # Best-crop recommendation per location × year
│
├── outputs/
│   ├── figures/                      # All plots from all notebooks
│   ├── models/                       # Trained LightGBM models (.joblib)
│   ├── baseline_metrics.json         # RMSE, MAE, R² for full-season model
│   ├── early_prediction_metrics.json # Metrics for each day-cutoff model
│   ├── baseline_predictions.parquet  # Test set predictions (full-season)
│   ├── crop_recommendations.parquet  # Best crop per location × year
│   └── feature_importance.csv        # Feature importance ranking
│
└── src/
    └── data/
        └── loader.py                 # DuckDB helpers for reading raw parquet
```

## Notebooks

### 01 — EDA
Explores yield distribution, water scenario effects, spatial patterns, drought sensitivity, and feature correlations. Uses DuckDB to query the 5 GB parquet without loading it into RAM.

### 02 — Feature Engineering
Aggregates hourly data to season-level (one row per location × crop × year × scenario). Applies year-based train/test split (2023–2024 as test) and target-encodes `crop_name`.

### 03 — Baseline Model
Trains a LightGBM regressor on full-season features.

| Metric | Value |
|---|---|
| R² | 0.964 |
| RMSE | 553 kg/ha |
| MAE | — |

Top features: `max_tagp`, `max_lai`, `mean_rftra`, `max_dvs`.

### 04 — Early Prediction
Trains separate models using only the first N days of each season to simulate real-world early-warning scenarios.

| Day cutoff | R² | RMSE (kg/ha) |
|---|---|---|
| 30 | 0.717 | 1,545 |
| 60 | 0.753 | 1,444 |
| 90 | 0.789 | 1,334 |
| 120 | 0.791 | 1,328 |
| Full season | 0.964 | 553 |

**Key finding:** Performance plateaus after day 90 — this is the practical early-warning cutoff.

### 05 — Crop Recommendation
Uses the full-season model to predict yield for all crops at each location and recommends the highest-yielding crop. Normal scenario only.

## Setup

```bash
pip install -r requirements.txt
```

Run notebooks in order: `01 → 02 → 03 → 04 → 05`

> **Note:** `02` must be run before `03–05` since it generates `train.parquet`, `test.parquet`, and `crop_te_map.json`.
