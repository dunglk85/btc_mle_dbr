---
title: 'BTC MLOps - GitHub Repo Setup'
type: 'feature'
created: '2026-05-30'
status: 'done'
route: 'one-shot'
---

## Intent

**Problem:** Dự án BTC Databricks MLOps cần một GitHub repository với cấu trúc thư mục chuẩn, CI/CD pipeline, và skeleton code để bắt đầu implementation.

**Approach:** Tạo Git repo với cấu trúc thư mục notebook-first rõ ràng (notebooks/, docs/, databricks/, .github/workflows/), kèm CI/CD workflows và DABs config.

## Suggested Review Order

**CI/CD & Deployment**

- CI pipeline: lint, unit test, bundle validation
  [`ci.yml`](../../.github/workflows/ci.yml)

- CD pipeline: deploy dev → prod via DABs
  [`cd.yml`](../../.github/workflows/cd.yml)

**Databricks Infrastructure**

- DABs bundle config with dev/prod targets
  [`databricks.yml`](../../databricks/databricks.yml)

- Job definitions for ingestion & retrain pipelines
  [`jobs.yml`](../../databricks/resources/jobs.yml)

**Notebook Layer**

- Binance ingestion, feature engineering, model training, prediction, and monitoring live under [`notebooks/`](../../notebooks/).

**Validation**

- CI validates notebook syntax with `python -m py_compile` and validates Databricks Asset Bundles.
