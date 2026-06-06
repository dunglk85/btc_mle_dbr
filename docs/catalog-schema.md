# Catalog Schema

## Multi-Environment Catalogs

| Catalog    | Purpose      |
|------------|-------------|
| `btc_dev`  | Development  |
| `btc_stg`  | Staging      |
| `btc_prod` | Production   |

## Schema Structure (per catalog)

```
<catalog>/
├── raw/
│   ├── landing           # UC Volume for landing CSV files plus Auto Loader checkpoints/schemas
│   ├── btc_hourly_landing_autoloader # Auto Loader staging Delta table
│   └── btc_hourly        # Raw OHLCV Delta table from Binance
├── features/
│   └── btc_features      # Engineered features
├── predictions/
│   └── btc_predictions   # Model predictions
├── monitoring/
│   ├── pipeline_metrics
│   ├── model_refresh_decisions
│   ├── training_dataset_manifests
│   └── data_remediation_actions
└── models/
    └── btc_price_model   # UC registered model with Champion/Challenger aliases
```

## btc_hourly Schema

Auto Loader state for this table is stored in the `raw.landing` volume:

- Checkpoint: `/Volumes/<catalog>/raw/landing/_checkpoints/btc_hourly`
- Schema tracking: `/Volumes/<catalog>/raw/landing/_schemas/btc_hourly`

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

## Monitoring Tables

`pipeline_metrics` records pipeline health metrics with `metric_time`, `metric_name`, `metric_value`, `status`, and `details`.

`model_refresh_decisions` records whether the model refresh job should train, including the reason, latest raw freshness, alert count, Champion existence, table versions, and JSON decision details.

`training_dataset_manifests` records the raw/features/config Delta versions, feature config ID, feature columns, split boundaries, and row counts used for each MLflow training run.

`data_remediation_actions` records safe auto-remediation attempts, blocked manual-required cases, and failure reasons.

`predictions.btc_predictions` includes model/data lineage fields: `model_version`, `model_run_id`, prediction-input `raw_table_version`/`features_table_version`, and Champion training lineage fields `model_raw_table_version`, `model_features_table_version`, `model_feature_config_version`, `model_feature_config_id`.
