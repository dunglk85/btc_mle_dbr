# BTC Databricks MLOps

End-to-end MLOps system on Databricks for Bitcoin price prediction (hourly time series).

## Architecture

- **Data**: Binance Vision API -> direct raw Delta MERGE (Unity Catalog)
- **ML**: active feature selection config + Optuna LightGBM/XGBoost training + MLflow tracking
- **CI/CD**: GitHub Actions + Databricks Asset Bundles (DABs)
- **Monitoring**: Data quality, model performance, job health

## Project Structure

```
├── .github/workflows/    # CI/CD pipelines
├── databricks/            # DABs configuration & job definitions
├── notebooks/             # Databricks notebooks
├── src/                   # Python source code
│   ├── data/              # Ingestion & feature engineering
│   ├── models/            # Training & evaluation
│   ├── monitoring/        # Data & model monitoring
│   └── utils/             # Config, logging
├── tests/                 # Unit tests
├── configs/               # Optuna search spaces
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
pytest
ruff check src/ tests/
databricks bundle validate
```

## Databricks Jobs

- `btc_data_prediction_job`: hourly ingestion, feature engineering, feature selection, LightGBM/XGBoost training, dataset replay, Champion/Challenger promotion, prediction, and monitoring.

## CI/CD Branch Mapping

- `simplying` branch deploys Databricks target `simplying` with Unity Catalog `btc_simply`.
- `main` branch deploys Databricks target `prod` with Unity Catalog `btc_prod`.
- Databricks Git-backed jobs use branch `simplying` for target `simplying` and branch `main` for target `prod`.
- Create the target catalog once in the Databricks UI using Default Storage before first deploy.
- CI/CD also creates required schemas: `raw`, `features`, `predictions`, `monitoring`, and `models`.

## Production-Like Controls

- Feature selection is governed by append-only `features.feature_selection_config` with one active config.
- Training logs raw/features/config Delta versions to MLflow and `monitoring.training_dataset_manifests`.
- `12_training_dataset_replay.py` validates `VERSION AS OF` replay before model promotion.
- Champion/Challenger promotion compares both models on the same bounded holdout rows.
- Predictions store both serving-input lineage and Champion training lineage.

Dashboard SQL templates are in `databricks/sql/` and use a `catalog` parameter such as `btc_simply` or `btc_prod`.

SQL alerts are managed by DAB in `databricks/resources/alerts.yml`. Deploy them with `sql_warehouse_id` set:

```bash
databricks bundle deploy --var="sql_warehouse_id=<warehouse-id>"
```

The monitoring dashboard is managed by DAB in `databricks/resources/dashboards.yml` using the exported layout in `databricks/dashboards/`.

## License

MIT
