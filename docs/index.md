# Project Documentation Index

## Project Overview

- **Project:** BTC Databricks MLOps
- **Type:** Data/ML pipeline on Databricks
- **Primary language:** Python
- **Architecture:** Databricks Asset Bundles + Git-backed notebooks + direct Binance ingestion + Unity Catalog Delta tables + MLflow UC Model Registry

## Quick Reference

- **Simplying catalog:** `btc_simply`
- **Prod catalog:** `btc_prod`
- **Pipeline job:** `btc_data_prediction_job`, hourly full data/train/predict/monitor path
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
- `notebooks/01_data_ingestion.py`: fetch closed Binance hourly candles and MERGE directly into raw Delta.
- `notebooks/02_feature_engineering.py`: features and exact next-hour target.
- `notebooks/02b_eda_feature_selection.py`: EDA-driven active feature selection config governance.
- `notebooks/03_optuna_training.py`: regression Optuna LightGBM/XGBoost training, MLflow logging, and dataset manifest writing.
- `notebooks/04_champion_challenger.py`: replay-gated, bounded fair Champion/Challenger registration and alias promotion.
- `notebooks/05_prediction.py`: Champion prediction writes with serving-input and model-training lineage.
- `notebooks/06_monitoring.py`: pipeline metrics.
- `notebooks/07_training_gate.py`: optional training/retraining gate decisions for gated refresh flows.
- `notebooks/08_drift_monitoring.py`: data, label, prediction, model, and concept-drift proxy metrics.
- `notebooks/09_job_quality_monitoring.py`: Databricks job quality metrics.
- `notebooks/10_data_remediation.py`: optional safe auto-remediation for stale raw data, stale features, and stale predictions.
- `notebooks/11_trigger_model_refresh.py`: optional conditional trigger for separated model refresh flows.
- `notebooks/12_training_dataset_replay.py`: production-like replay validation for training dataset manifests before model promotion.
- `databricks/sql/`: dashboard and alert SQL templates.
- `databricks/resources/alerts.yml`: CI/CD-managed Databricks SQL alerts.
- `databricks/resources/dashboards.yml`: CI/CD-managed AI/BI dashboard resource.
- `databricks/dashboards/BTC MLOps Monitoring Dashboard.lvdash.json`: exported dashboard layout.

## Validation Commands

```bash
pytest
ruff check src/ tests/
python -m py_compile notebooks/01_data_ingestion.py notebooks/02_feature_engineering.py notebooks/02b_eda_feature_selection.py notebooks/03_model_training.py notebooks/03_optuna_training.py notebooks/04_champion_challenger.py notebooks/05_prediction.py notebooks/06_monitoring.py notebooks/07_training_gate.py notebooks/08_drift_monitoring.py notebooks/09_job_quality_monitoring.py notebooks/10_data_remediation.py notebooks/11_trigger_model_refresh.py notebooks/12_training_dataset_replay.py notebooks/13_select_best_challenger.py
databricks bundle validate
```
