# Project Documentation Index

## Project Overview

- **Project:** BTC Databricks MLOps
- **Type:** Data/ML pipeline on Databricks
- **Primary language:** Python
- **Architecture:** Databricks Asset Bundles + Git-backed notebooks + Unity Catalog Delta tables + MLflow UC Model Registry

## Quick Reference

- **Dev catalog:** `btc_dev`
- **Prod catalog:** `btc_prod`
- **Data job:** `btc_data_prediction_job`, hourly
- **Model refresh job:** `btc_model_refresh_job`, every 12 hours, paused by default
- **Registered model:** `<catalog>.models.btc_price_model`
- **Champion alias:** `@Champion`
- **Challenger alias:** `@Challenger`

## Documentation

- [README](../README.md)
- [Architecture](./architecture.md)
- [Catalog Schema](./catalog-schema.md)
- [Monitoring Dashboard And Alerts](./monitoring-dashboard.md)
- [Original Project Brief](./btc-databricks-mlops-project.md)
- [Technical Explanation](../_bmad-output/planning-artifacts/research/technical_explanation.md)
- [Technical Review](../_bmad-output/planning-artifacts/research/technical-review-btc-databricks-mlops-implementation-plan-research-2026-05-29.md)

## Key Code Areas

- `databricks.yml`: bundle targets and catalog variables.
- `databricks/resources/jobs.yml`: Databricks job definitions.
- `notebooks/00_fetch_binance_to_volume.py`: fetch closed Binance hourly candles into UC Volume.
- `notebooks/01_data_ingestion.py`: landing CSV to raw Delta MERGE.
- `notebooks/02_feature_engineering.py`: features and exact next-hour target.
- `notebooks/03_optuna_training.py`: Optuna training and MLflow logging.
- `notebooks/04_champion_challenger.py`: UC model registration and alias promotion.
- `notebooks/05_prediction.py`: Champion prediction writes.
- `notebooks/06_monitoring.py`: pipeline metrics.
- `notebooks/07_monitoring_gate.py`: model refresh decisions.
- `notebooks/08_drift_monitoring.py`: data, label, prediction, model, and concept-drift proxy metrics.
- `databricks/sql/`: dashboard and alert SQL templates.
- `databricks/resources/alerts.yml`: CI/CD-managed Databricks SQL alerts.
- `databricks/resources/dashboards.yml`: CI/CD-managed AI/BI dashboard resource.
- `databricks/dashboards/BTC MLOps Monitoring Dashboard.lvdash.json`: exported dashboard layout.

## Validation Commands

```bash
pytest
ruff check src/ tests/ scripts/
python -m py_compile notebooks/00_fetch_binance_to_volume.py notebooks/01_data_ingestion.py notebooks/02_feature_engineering.py notebooks/03_model_training.py notebooks/03_optuna_training.py notebooks/04_champion_challenger.py notebooks/05_prediction.py notebooks/06_monitoring.py notebooks/07_monitoring_gate.py
databricks bundle validate
```
