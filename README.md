# BTC Databricks MLOps

End-to-end MLOps system on Databricks for Bitcoin price prediction (hourly time series).

## Architecture

- **Data**: Binance API → Delta Lake (Unity Catalog)
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
| Production  | `btc_prd` | `prod`      |

## Quick Start

```bash
pip install -r requirements.txt
```

## License

MIT
