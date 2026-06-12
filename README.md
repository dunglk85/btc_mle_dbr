# BTC Databricks MLOps

End-to-end MLOps system on Databricks for Bitcoin price prediction (hourly time series).

## Architecture

- **Data**: Binance Vision API -> direct raw Delta MERGE (Unity Catalog)
- **ML**: hourly inference with on-demand Optuna LightGBM/XGBoost/Random Forest training + MLflow tracking
- **CI/CD**: GitHub Actions + Databricks Asset Bundles (DABs)
- **Monitoring**: Data quality, model performance, job health

## Project Structure

```
├── .github/workflows/    # CI/CD pipelines
├── databricks/            # DABs configuration & job definitions
├── notebooks/             # Databricks notebooks
└── docs/                  # Architecture & design docs
```

## Environments

| Environment | Catalog   | DABs Target |
|-------------|-----------|-------------|
| Simplying   | `btc_simply` | `simplying` |
| Production  | `btc_prod` | `prod`      |

## Quick Start

```bash
pip install -r requirements.txt
python -m py_compile notebooks/01_data_ingestion.py notebooks/02_feature_engineering.py notebooks/03_optuna_training.py notebooks/04_champion_challenger.py notebooks/05_prediction.py notebooks/06_monitoring.py notebooks/test_drift_thresholds.py
databricks bundle validate
```

## Databricks Jobs

- `btc_inference_job`: hourly ingestion, feature engineering, Champion prediction, monitoring, and conditional training (LGBM/XGB/RF + champion/challenger) triggered by drift threshold.

## CI/CD Branch Mapping

- `simplying` branch deploys Databricks target `simplying` with Unity Catalog `btc_simply`.
- `main` branch deploys Databricks target `prod` with Unity Catalog `btc_prod`.
- Databricks Git-backed jobs use branch `simplying` for target `simplying` and branch `main` for target `prod`.
- Create the target catalog once in the Databricks UI using Default Storage before first deploy.
- CI/CD also creates required schemas: `raw`, `features`, `predictions`, `monitoring`, and `models`.

## Production-Like Controls

- Feature selection is governed by append-only `features.feature_selection_config` with one active config.
- Training logs raw/features/config Delta versions to MLflow and `monitoring.training_dataset_manifests`.
- Champion/Challenger promotion compares both models on the same bounded holdout rows.
- Predictions store both serving-input lineage and Champion training lineage.

Dashboard SQL templates are in `databricks/sql/` and use a `catalog` parameter such as `btc_simply` or `btc_prod`.

SQL alerts are managed by DAB in `databricks/resources/alerts.yml`. Deploy them with `sql_warehouse_id` set:

```bash
databricks bundle deploy --var="sql_warehouse_id=<warehouse-id>"
```

The monitoring dashboard is managed by DAB in `databricks/dashboards/BTC MLOps Monitoring Dashboard.lvdash.json`.

## License

MIT
