# BTC Databricks MLOps

End-to-end MLOps system on Databricks for Bitcoin price prediction (hourly time series).

## Architecture

- **Data**: Binance Vision API -> UC Volume landing CSV -> Auto Loader -> Delta Lake (Unity Catalog)
- **ML**: Optuna hyperparameter tuning + MLflow tracking
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
| Dev         | `btc_dev` | `dev`       |
| Production  | `btc_prod` | `prod`      |

## Quick Start

```bash
pip install -r requirements.txt
pytest
ruff check src/ tests/ scripts/
databricks bundle validate
```

## Databricks Jobs

- `btc_data_prediction_job`: hourly fetch, ingestion, feature engineering, prediction, monitoring.
- `btc_drift_monitoring_job`: drift metrics and drift gate checks every 6 hours.
- `btc_model_refresh_job`: monitoring gate, Optuna training, Champion/Challenger registration every 12 hours.

Dashboard SQL templates are in `databricks/sql/` and use a `catalog` parameter such as `btc_dev` or `btc_prod`.

SQL alerts are managed by DAB in `databricks/resources/alerts.yml`. Deploy them with `sql_warehouse_id` set:

```bash
databricks bundle deploy --var="sql_warehouse_id=<warehouse-id>"
```

The monitoring dashboard is managed by DAB in `databricks/resources/dashboards.yml` using the exported layout in `databricks/dashboards/`.

## License

MIT
