# Catalog Schema

## Multi-Environment Catalogs

| Catalog    | Purpose      |
|------------|-------------|
| `btc_simply`  | Development  |
| `btc_stg`  | Staging      |
| `btc_prod` | Production   |

## Schema Structure (per catalog)

```
<catalog>/
├── raw/
│   └── btc_hourly        # Raw OHLCV Delta table from Binance
├── features/
│   ├── btc_features      # Engineered features
│   └── feature_selection_config # Append-only active feature selection metadata
├── predictions/
│   └── btc_predictions   # Model predictions
├── monitoring/
│   ├── pipeline_metrics
│   └── training_dataset_manifests
└── models/
    └── btc_price_model   # UC registered model with Champion/Challenger aliases
```

## btc_hourly Schema

`01_data_ingestion` fetches closed Binance hourly candles and MERGEs them directly into this table. No UC Volume landing path or Auto Loader staging table is used.

| Column        | Type      | Description                |
|---------------|-----------|----------------------------|
| open_time     | timestamp | Start of candle            |
| open          | double    | Open price                 |
| high          | double    | High price                 |
| low           | double    | Low price                  |
| close         | double    | Close price (target)       |
| volume        | double    | Volume                     |
| close_time    | timestamp | End of candle              |
| quote_volume  | double    | Quote asset volume         |
| trades        | bigint    | Number of trades           |

## btc_features Notes

- `target_close_1h` is the exact close price for `open_time + 1 hour`.
- If the next hourly candle is missing, `target_close_1h` is null.
- Training drops rows with null feature or target values.

## Feature Selection Config

`features.feature_selection_config` is append-only governance metadata for selected model features.

Important fields:

- `config_key`: currently `selected_features`.
- `config_id` / `config_version`: timestamp-derived config identifier.
- `is_active`: only one active selected-features config should exist after `02_feature_engineering` runs.
- `source_table`, `source_table_version`: feature Delta table snapshot used for selection.
- `target_col`: regression target used for selection.
- `candidate_features_json`, `dropped_features_json`, `selection_metrics_json`: audit context for why the feature set was chosen.

Training reads only `is_active = true` configs unless `allow_default_feature_fallback=true` is explicitly set.

## Monitoring Tables

`pipeline_metrics` records pipeline health metrics with `metric_time`, `metric_name`, `metric_value`, `status`, and `details`.

`training_dataset_manifests` records the raw/features/config Delta versions, feature config ID, feature columns, split boundaries, and row counts used for each MLflow training run.

`predictions.btc_predictions` includes model/data lineage fields: `model_version`, `model_run_id`, prediction-input `raw_table_version`/`features_table_version`, and Champion training lineage fields `model_raw_table_version`, `model_features_table_version`, `model_feature_config_version`, `model_feature_config_id`.

Important prediction columns:

- `feature_open_time`: feature row timestamp used for inference.
- `predicted_close`: next-hour close forecast used by dashboard/error monitoring.
- `predicted_return_1h`: raw return forecast when Champion target is `target_return_1h`; derived from close forecast otherwise.
- `model_target_col`: Champion training target, currently expected to be `target_return_1h` for regression runs.
- `raw_table_version`, `features_table_version`: serving input table versions at prediction time.
- `model_raw_table_version`, `model_features_table_version`, `model_feature_config_version`, `model_feature_config_id`: Champion training lineage.
