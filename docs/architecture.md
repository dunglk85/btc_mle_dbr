# Architecture

## Data Flow

1. **Binance Vision API** -> `00_fetch_binance_to_volume` -> UC Volume landing CSV.
2. **Auto Loader ingestion** -> `01_data_ingestion` incrementally reads new landing files into `<catalog>.raw.btc_hourly_landing_autoloader`, then batch MERGEs into `<catalog>.raw.btc_hourly`.
3. **Feature Engineering** -> `<catalog>.features.btc_features` with exact next-hour target `target_close_1h`.
4. **Model Training** -> Optuna RandomForest + MLflow tracking.
5. **Champion vs Challenger** -> Register current training run as Challenger, compare RMSE, promote winner.
6. **Prediction** -> `<catalog>.predictions.btc_predictions` using `@Champion`.
7. **Monitoring** -> `<catalog>.monitoring.pipeline_metrics` and model refresh gate decisions.

## Multi-Environment

| Environment | Unity Catalog | DABs Target |
|-------------|---------------|-------------|
| Dev         | `btc_dev`     | `dev`       |
| Production  | `btc_prod`    | `prod`      |

## Schedules

- **Data prediction job**: every hour.
- **Model refresh job**: every 12 hours, paused by default.

## Environment Parameterization

Databricks notebooks read the `catalog` widget passed by Databricks Asset Bundles. The default is `btc_dev`; the prod target passes `btc_prod`.

## Data Correctness Rules

- Fetch excludes currently open candles by requiring Binance `close_time` to be before current UTC time.
- Feature target is an exact one-hour lookup, not just the next available row.
- Auto Loader tracks processed landing files with a checkpoint under `/Volumes/<catalog>/raw/landing/_checkpoints/btc_hourly`.
- Ingestion deduplicates overlapping landing files deterministically using Unity Catalog `_metadata.file_path`.

## Drift Monitoring Status

Current monitoring is operational fallback monitoring, not full statistical drift detection.

Implemented now:
- Raw freshness.
- Raw duplicate/null timestamp checks.
- Feature row count and target null checks.
- Prediction availability and age.
- Actual-vs-predicted SQL queries for dashboard/alerts.
- Job quality metrics, including success rate, failed runs, failed tasks, and latest run duration.

Implemented drift monitoring:
- `notebooks/08_drift_monitoring.py` writes drift metrics into `<catalog>.monitoring.pipeline_metrics`.
- Data drift: PSI and approximate KS for selected features.
- Label drift: PSI/KS for `target_close_1h`.
- Prediction drift: PSI/KS for `predicted_close`.
- Model/performance drift: rolling RMSE, MAE, MAPE, p95 absolute error, direction accuracy.
- Concept drift proxy: rolling signed error bias.

Drift alerts influence retraining:
- Drift alerts are retraining candidates, not automatic retraining approval.
- The gate validates data quality, schema quality, and feature quality before retraining.
- Blocking quality/schema/feature alerts stop retraining.
- Drift alerts trigger retraining only when validation passes.

Retraining decision flow:

```text
Data drift / prediction drift / feature drift alert
        ↓
Validate data quality + schema quality + feature quality
        ↓
If validation passes: retrain
If validation fails: block retrain and alert operator
```

Immediate drift-triggered retraining:
- `btc_data_prediction_job` runs `drift_monitoring` after regular monitoring.
- It also runs `job_quality_monitoring` to record Databricks job health metrics.
- It then runs `monitoring_gate_drift` with `trigger_mode=drift`.
- If drift exists and validation passes, `model_training_drift` runs immediately.
- If no drift exists, `model_training_drift` exits with `SKIP_RETRAIN`.
