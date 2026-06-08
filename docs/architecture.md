# Architecture

## Data Flow

1. **Direct Binance ingestion** -> `01_data_ingestion` fetches closed BTC hourly candles from Binance Vision API and MERGEs them into `<catalog>.raw.btc_hourly`.
2. **Feature Engineering** -> `<catalog>.features.btc_features` with exact next-hour target `target_close_1h`.
3. **Feature Selection Config** -> `02b_eda_feature_selection` writes append-only active selected-feature metadata into `<catalog>.features.feature_selection_config`.
4. **Model Training** -> Regression-only Optuna LightGBM/XGBoost training + MLflow tracking.
5. **Dataset Replay Validation** -> `12_training_dataset_replay` validates Delta `VERSION AS OF` reproducibility from `training_dataset_manifests`.
6. **Champion vs Challenger** -> Register current training run as Challenger, evaluate Challenger and current Champion on the same bounded holdout rows, then promote only if RMSE and MAE improve and directional accuracy does not regress.
7. **Prediction** -> `<catalog>.predictions.btc_predictions` using `@Champion`; return forecasts are converted to `predicted_close` for monitoring.
8. **Monitoring** -> `<catalog>.monitoring.pipeline_metrics` and model refresh gate decisions.

## Multi-Environment

| Environment | Unity Catalog | DABs Target |
|-------------|---------------|-------------|
| Dev         | `btc_dev`     | `dev`       |
| Production  | `btc_prod`    | `prod`      |

## Schedules

- **Data prediction job**: every hour.
- **Drift monitoring job**: every 6 hours.
- **Model refresh job**: trigger-only from drift monitoring decisions.

## Environment Parameterization

Databricks notebooks read the `catalog` widget passed by Databricks Asset Bundles. The default is `btc_dev`; the prod target passes `btc_prod`.

## Data Correctness Rules

- Fetch excludes currently open candles by requiring Binance `close_time` to be before current UTC time.
- Feature target is an exact one-hour lookup, not just the next available row.
- Feature selection config is append-only with one active config; each config records source feature table version, target column, candidates, dropped features, and selection metrics.
- Ingestion reads from the latest raw `open_time` by default and can backfill from a `start_date` widget.
- Ingestion deduplicates overlapping Binance candles by `open_time` before MERGE.
- Training logs Delta versions for raw/features/config tables into MLflow and `monitoring.training_dataset_manifests`.
- `12_training_dataset_replay` validates that manifest versions are still available with Delta time travel and that the replayed training dataset matches the manifest before model promotion.
- Champion/Challenger evaluation uses a bounded latest common holdout window so both models are compared on identical rows without loading the full feature table.
- Predictions store model version/run ID, prediction-input raw/features Delta versions, and Champion training data/config versions for traceability.

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
- Blocking data-quality alerts trigger safe data remediation where possible, not retraining.

Retraining decision flow:

```text
Data drift / prediction drift / feature drift alert
        ↓
Validate data quality + schema quality + feature quality
        ↓
If validation passes: retrain
If validation fails: block retrain and trigger remediation/manual review
```

Job separation:
- `btc_data_prediction_job` runs only the hourly serving path: ingestion, feature engineering, prediction, regular monitoring, and job quality monitoring.
- `btc_drift_monitoring_job` runs `drift_monitoring`, `training_gate_drift`, conditional model-refresh trigger, and safe data remediation every 6 hours.
- `btc_model_refresh_job` owns trigger-only feature selection, retraining, dataset replay validation, and Champion/Challenger promotion; training notebooks require a fresh positive training-gate decision. LightGBM and XGBoost training can run independently, but promotion is serialized to avoid alias races.
