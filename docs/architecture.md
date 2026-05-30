# Architecture

## Data Flow

1. **Binance API** → Ingestion job (hourly) → `btc_dev.raw.btc_hourly` (Delta)
2. **Feature Engineering** → `btc_dev.features.btc_features`
3. **Model Training** (Optuna tuning + MLflow tracking)
4. **Champion vs Challenger** → Compare on test set → Promote winner
5. **Prediction** → `btc_dev.predictions.btc_predictions`

## Multi-Environment

| Environment | Unity Catalog | DABs Target |
|-------------|---------------|-------------|
| Dev         | `btc_dev`     | `dev`       |
| Production  | `btc_prd`     | `prod`      |

## Schedules

- **Ingestion**: every hour
- **Retrain**: every 3 hours (configurable)
