# BTC MLOps Drift Metrics & Drift Checking Documentation

## Overview

The BTC MLOps pipeline uses `notebooks/06_monitoring.py` to compute drift metrics by comparing two time windows:
- **Recent window**: latest `recent_hours` (default 168 hours = 7 days)
- **Reference window**: previous `reference_hours` (default 720 hours = 30 days)

All metrics are written to `{catalog}.monitoring.pipeline_metrics` with fields: `metric_time`, `metric_name`, `metric_value`, `status` (ok/warn/alert), `details`.

---

## Statistical Tests Used

### PSI (Population Stability Index)

**How calculated:**
1. Build quantile buckets from the reference window (10 buckets by default)
2. Count percentage of reference rows in each bucket
3. Count percentage of recent rows in each bucket
4. Sum: `(recent_pct - ref_pct) * ln(recent_pct / ref_pct)`

**Thresholds:**
| Status | PSI Value |
|--------|-----------|
| ok | < 0.30 |
| warn | 0.30 - 1.5 |
| alert | >= 1.5 |

### Approximate KS (Kolmogorov-Smirnov)

**How calculated:**
1. Compute reference quantiles at 10%, 20%, ..., 90%
2. Estimate CDF difference between reference and recent at those quantiles
3. Return maximum CDF difference

**Thresholds:**
| Status | KS Value |
|--------|----------|
| ok | < 0.30 |
| warn | 0.30 - 0.60 |
| alert | >= 0.60 |

---

## Drift Metric Categories

### 1. Data Drift Metrics

**Purpose:** Detect if input feature distribution changed compared to reference window.

**Features monitored:** `return_1h`

**Metric names:**
- `data_drift_psi_return_1h`
- `data_drift_ks_return_1h`

**Note:** The test notebook (`test_drift_thresholds.py`) validates thresholds against `volume`, `quote_volume`, `trades`, `return_1h`, but production monitoring currently only tracks `return_1h` for data drift retrain triggers.

---

### 2. Schema Drift Metrics

**Purpose:** Detect if expected feature columns are missing from the feature table.

**Metric names:**
- `schema_drift_missing_<feature>` (e.g., `schema_drift_missing_return_1h`)

**Status:** Always `alert` when emitted.

---

### 3. Feature Quality Metrics

**Purpose:** Detect if feature data quality degraded (null rates).

**Metric names:**
- `feature_quality_null_rate_<feature>`

**Thresholds:**
| Status | Null Rate |
|--------|-----------|
| ok | < 5% |
| warn | 5% - 20% |
| alert | >= 20% |

---

### 4. Label Drift Metrics (Monitor-Only)

**Purpose:** Detect if target distribution changed.

**Metric names:**
- `label_drift_psi_target_close_1h`
- `label_drift_ks_target_close_1h`

**Behavior:** Monitor-only. Uses effectively infinite alert thresholds (`psi_monitor_only_alert_threshold = 999.0`, `ks_monitor_only_alert_threshold = 999.0`) so it can produce `warn` but never `alert`. This avoids retraining every time BTC price naturally trends.

---

### 5. Price Level Drift Metrics (Monitor-Only)

**Purpose:** Track BTC price-level distribution shifts without triggering retraining.

**Columns tracked:** `close`, `ma_24`

**Metric names:**
- `price_level_drift_psi_close`
- `price_level_drift_ks_close`
- `price_level_drift_psi_ma_24`
- `price_level_drift_ks_ma_24`

**Behavior:** Monitor-only with infinite alert thresholds. BTC price naturally trends, so these would constantly alert if used as retrain triggers.

---

### 6. Prediction Drift Metrics (Monitor-Only)

**Purpose:** Detect if model output distribution changed (before actuals are available).

**Metric names:**
- `prediction_drift_psi_predicted_close`
- `prediction_drift_ks_predicted_close`

**Behavior:** Monitor-only with infinite alert thresholds. Prediction drift can happen before actual labels are available, so it's tracked separately from model performance drift.

---

### 7. Model / Performance Drift Metrics

**Purpose:** Detect if model prediction quality degraded.

**Join logic:** `predictions.feature_open_time + INTERVAL 1 HOUR = raw.open_time`

**Metrics:**

| Metric Name | Description | Formula | Thresholds |
|-------------|-------------|---------|------------|
| `model_drift_joined_prediction_count` | Count of predictions joinable with actuals | - | ok (always) |
| `model_drift_rmse_{recent_hours}h` | Root Mean Squared Error | `sqrt(avg((actual - predicted)^2))` | ok (always) |
| `model_drift_mae_{recent_hours}h` | Mean Absolute Error | `avg(\|actual - predicted\|)` | ok (always) |
| `model_drift_r2_{recent_hours}h` | R-squared | `1 - SS_res/SS_tot` | ok (always) |
| `model_drift_mape_{recent_hours}h` | Mean Absolute % Error | `avg(\|error\| / \|actual\|)` | warn >= 0.02, alert >= 0.05 |
| `model_drift_p95_abs_error_{recent_hours}h` | 95th percentile absolute error | - | ok (always) |
| `model_drift_direction_accuracy_{recent_hours}h` | Direction match rate | `avg(predicted_dir == actual_dir)` | warn <= 0.48, alert <= 0.45 |

**Note:** `{recent_hours}` defaults to 168 in the metric name.

---

### 8. Concept Drift Proxy Metric

**Purpose:** Detect if the relationship between features and target changed.

**Metric name:**
- `concept_drift_mean_error_bias_{recent_hours}h`

**Formula:** `avg(actual_close - predicted_close)`

**Interpretation:**
- Positive persistent bias = model tends to underpredict
- Negative persistent bias = model tends to overpredict
- This is a proxy metric, not a full causal concept drift test

---

## Drift Checking in Jobs

### Drift Alert Counting

After all metrics are computed, `06_monitoring.py` counts drift alerts:

```python
drift_alert_count = spark.createDataFrame(metrics).filter(
    F.col("status") == "alert"
).filter(
    F.col("metric_name").rlike("^(data_drift|label_drift|prediction_drift|model_drift|concept_drift)_")
).count()
```

This counts alerts from these prefixes only:
- `data_drift_*`
- `label_drift_*`
- `prediction_drift_*`
- `model_drift_*`
- `concept_drift_*`

**Excluded from drift alert count:**
- `price_level_drift_*` (monitor-only)
- `schema_drift_*` (blocking quality issue, not drift trigger)
- `feature_quality_*` (blocking quality issue, not drift trigger)

### Training Gate Decision

The `drift_alert_count` is compared against `min_drift_alerts_to_trigger` (default: 2):

```
drift_threshold_met = drift_alert_count >= min_drift_alerts_to_trigger
```

The count is stored as a job task value: `dbutils.jobs.taskValues.set(key="drift_alert_count", value=drift_alert_count)`

### Retraining Decision Flow

```
Model drift / concept drift alert
        ↓
Validate data quality + schema quality + feature quality
        ↓
If validation passes: should_retrain = true
If validation fails: should_retrain = false
```

**Blocking alert types** (prevent retraining):
- `raw_freshness_hours`
- `raw_duplicate_open_time_count`
- `raw_null_open_time_count`
- `features_target_close_1h_null_count`
- `raw_features_row_count_delta`
- `feature_quality_*`
- `schema_drift_*`

**Retraining trigger alert types:**
- `data_drift_*`
- `model_drift_*`
- `prediction_drift_*`
- `label_drift_*`
- `concept_drift_*`

**Monitor-only drift types** (shown on dashboard, never trigger retraining):
- `price_level_drift_*`

---

## SQL Alert Queries

Defined in `databricks/sql/alert_queries.sql`:

| Alert # | Query | Condition |
|---------|-------|-----------|
| 1 | Raw Data Stale | `raw_freshness_hours > 3` |
| 2 | Monitoring Has Alerts | `alert_count > 0` (last 2 hours) |
| 3 | No Recent Prediction | `prediction_age_hours > 3` |
| 4 | High Prediction Error | `avg_pct_error > 0.05` (last 24h) |
| 5 | Feature Table Missing Targets | `target_null_count > 1` |
| 6 | Any Drift Alert | `drift_alert_count > 0` (last 2h, drift prefixes) |
| 7 | Feature Quality / Schema Alert | `quality_alert_count > 0` (last 2h) |
| 8 | Training Trigger Failure | `training_trigger_failure_count > 0` (last 2h) |

---

## Dashboard Queries

Defined in `databricks/sql/dashboard_queries.sql`:

| Query # | Purpose |
|---------|---------|
| 1 | Data Freshness Tile |
| 2 | Raw Data Count Tile |
| 3 | Feature Table Count Tile |
| 4 | Prediction Count Tile |
| 5 | Latest Predictions Table |
| 6 | Actual vs Predicted Table |
| 7 | Prediction Error Trend |
| 7b | Prediction Debug (legacy return-as-close rows) |
| 8 | Monitoring Alerts Table (alert/warn only) |
| 9 | Monitoring Metrics Timeline |
| 10 | Latest Drift Metrics (all drift prefixes) |
| 11 | Drift Alerts And Warnings |
| 12 | BTC Trading Volume Trend |
| 13 | Latest Model SHAP Explanation |
| 14 | Latest Model Feature Importance |

---

## Threshold Validation

The `test_drift_thresholds.py` notebook validates PSI/KS thresholds against historical data by testing:

**PSI thresholds tested:** 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 1.0
**KS thresholds tested:** 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6

**Window configurations:**
| Label | Recent | Reference |
|-------|--------|-----------|
| 24h vs 7d | 24h | 168h |
| 48h vs 14d | 48h | 336h |
| 168h vs 30d | 168h | 720h |
| 336h vs 60d | 336h | 1440h |

**Target:** ~10-30% alert rate (not too noisy, not too silent).

---

## Key Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `recent_hours` | 168 | Recent window size (7 days) |
| `reference_hours` | 720 | Reference window size (30 days) |
| `psi_warn_threshold` | 0.30 | PSI warning threshold |
| `psi_alert_threshold` | 1.5 | PSI alert threshold |
| `ks_warn_threshold` | 0.30 | KS warning threshold |
| `ks_alert_threshold` | 0.60 | KS alert threshold |
| `psi_monitor_only_alert_threshold` | 999.0 | Effectively infinite for monitor-only metrics |
| `ks_monitor_only_alert_threshold` | 999.0 | Effectively infinite for monitor-only metrics |
| `mape_warn_threshold` | 0.02 | MAPE warning (2%) |
| `mape_alert_threshold` | 0.05 | MAPE alert (5%) |
| `direction_warn_threshold` | 0.48 | Direction accuracy warning (48%) |
| `direction_alert_threshold` | 0.45 | Direction accuracy alert (45%) |
| `min_drift_alerts_to_trigger` | 2 | Minimum drift alerts to trigger retraining |
| `fail_on_alert` | false | Whether to raise exception on alerts |
| `expected_feature_lookback_loss` | 168 | Expected row loss between raw and features |
