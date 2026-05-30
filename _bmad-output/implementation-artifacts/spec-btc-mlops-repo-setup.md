---
title: 'BTC MLOps - GitHub Repo Setup'
type: 'feature'
created: '2026-05-30'
status: 'done'
route: 'one-shot'
---

## Intent

**Problem:** Dự án BTC Databricks MLOps cần một GitHub repository với cấu trúc thư mục chuẩn, CI/CD pipeline, và skeleton code để bắt đầu implementation.

**Approach:** Tạo Git repo với cấu trúc thư mục rõ ràng (notebooks/, src/, tests/, configs/, docs/, databricks/, .github/workflows/), kèm theo skeleton Python source code cho ingestion, feature engineering, training, monitoring, cùng CI/CD workflows và DABs config.

## Suggested Review Order

**CI/CD & Deployment**

- CI pipeline: lint, unit test, bundle validation
  [`ci.yml`](../../.github/workflows/ci.yml)

- CD pipeline: deploy dev → prod via DABs
  [`cd.yml`](../../.github/workflows/cd.yml)

- Hourly ingestion trigger via Databricks CLI
  [`hourly-trigger.yml`](../../.github/workflows/hourly-trigger.yml)

**Databricks Infrastructure**

- DABs bundle config with dev/prod targets
  [`databricks.yml`](../../databricks/databricks.yml)

- Job definitions for ingestion & retrain pipelines
  [`jobs.yml`](../../databricks/resources/jobs.yml)

**Data Layer**

- Binance API client with pagination + Delta merge upsert
  [`ingestion.py`](../../src/data/ingestion.py)

- Feature engineering: moving averages, lag features, time-based
  [`features.py`](../../src/data/features.py)

**Model Layer**

- Optuna hyperparameter tuning with MLflow tracking
  [`training.py`](../../src/models/training.py)

- Champion vs Challenger evaluation with auto-promote
  [`evaluation.py`](../../src/models/evaluation.py)

**Monitoring & Utilities**

- Data quality checks: missing values, freshness, schema validation
  [`data_quality.py`](../../src/monitoring/data_quality.py)

- Model performance tracking with MLflow logging
  [`model_performance.py`](../../src/monitoring/model_performance.py)

- Environment-aware structured logging (JSON/console)
  [`logger.py`](../../src/utils/logger.py)

- Config loader with env vars and YAML overrides
  [`config.py`](../../src/utils/config.py)

**Tests**

- Unit test for klines parsing
  [`test_ingestion.py`](../../tests/test_ingestion.py)

- Integration test for feature computation with PySpark
  [`test_features.py`](../../tests/test_features.py)

- Unit test for model evaluation metrics
  [`test_training.py`](../../tests/test_training.py)
