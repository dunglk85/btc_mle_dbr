# Project Documentation Index

## Project Overview

- **Project:** BTC Databricks MLOps
- **Type:** Data/ML pipeline on Databricks
- **Primary language:** Python
- **Architecture:** Databricks Asset Bundles + Git-backed notebooks + direct Binance ingestion + Unity Catalog Delta tables + MLflow UC Model Registry

## Quick Reference

- **Simplying catalog:** `btc_simply`
- **Prod catalog:** `btc_prod`
- **Inference job:** `btc_inference_job`, hourly ingest/features/predict/monitor/conditional-training path
- **Training tasks:** conditional within inference job, triggered by `check_drift_threshold` condition task
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
- `notebooks/02_feature_engineering.py`: features, exact next-hour target, and active selected-feature config governance.
- `notebooks/03_optuna_training.py`: regression Optuna LightGBM/XGBoost/Random Forest training, best-challenger selection, MLflow logging, and dataset manifest writing.
- `notebooks/04_champion_challenger.py`: best-candidate selection, bounded fair Champion/Challenger registration, and alias promotion.
- `notebooks/05_prediction.py`: Champion prediction writes with serving-input and model-training lineage.
- `notebooks/06_monitoring.py`: pipeline and drift metrics, sets `drift_alert_count` task value for conditional training gate.
- `notebooks/test_drift_thresholds.py`: manual historical PSI/KS threshold validation notebook.
- `databricks/sql/`: dashboard and alert SQL templates.
- `databricks/resources/alerts.yml`: CI/CD-managed Databricks SQL alerts.
- `databricks/dashboards/BTC MLOps Monitoring Dashboard.lvdash.json`: exported AI/BI dashboard layout with Model Performance & Explainability page.

## Validation Commands

```bash
python -m py_compile notebooks/01_data_ingestion.py notebooks/02_feature_engineering.py notebooks/03_optuna_training.py notebooks/04_champion_challenger.py notebooks/05_prediction.py notebooks/06_monitoring.py notebooks/test_drift_thresholds.py
databricks bundle validate
```
