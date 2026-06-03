# BTC MLOps Monitoring Dashboard And Alerts

## Prerequisites

Run the data prediction job at least once:

```text
btc_data_prediction_job
```

Confirm monitoring data exists:

```sql
SELECT COUNT(*) FROM IDENTIFIER(:catalog || '.monitoring.pipeline_metrics');
```

Confirm prediction table exists after a Champion model is available:

```sql
SELECT COUNT(*) FROM IDENTIFIER(:catalog || '.predictions.btc_predictions');
```

Set query parameter `catalog` to `btc_dev` or `btc_prod`.

## Dashboard Queries

Use the SQL queries in:

```text
databricks/sql/dashboard_queries.sql
```

Create a query/dashboard parameter named `catalog` and set it to `btc_dev` or `btc_prod`.

Recommended dashboard tiles:
- Data freshness: latest `raw_freshness_hours`.
- Raw count: latest `raw_count`.
- Feature count: latest `features_count`.
- Prediction count: latest `prediction_count`.
- Latest predictions table.
- Actual vs predicted table.
- Prediction error trend chart.
- Model refresh decisions table.
- Monitoring alerts table.

## SQL Alerts

Use the SQL queries in:

```text
databricks/sql/alert_queries.sql
```

Create an alert query parameter named `catalog` and set it to `btc_dev` or `btc_prod`.

Recommended alert conditions:
- Raw data stale: `raw_freshness_hours > 3`.
- Monitoring has alerts: `alert_count > 0`.
- No recent prediction: `prediction_age_hours > 3`.
- High prediction error: `avg_pct_error > 0.02`.
- Feature target nulls beyond expected last row: `target_null_count > 1`.

## Fallback If SQL Alerts Are Not Available

If Databricks SQL Alerts are unavailable in the current workspace, rely on job failure notifications.

The notebook:

```text
notebooks/06_monitoring.py
```

raises an error when alert metrics are produced. Configure job notifications on:

```text
btc_data_prediction_job
```

Notify on:
- Job failure.
- Task failure.

Recommended notification target:
- Email for now.
- Slack/webhook later if workspace supports it.

## Operational Checks

Latest monitoring metrics:

```sql
SELECT *
FROM IDENTIFIER(:catalog || '.monitoring.pipeline_metrics')
ORDER BY metric_time DESC
LIMIT 50;
```

Latest alert/warn metrics:

```sql
SELECT *
FROM IDENTIFIER(:catalog || '.monitoring.pipeline_metrics')
WHERE status IN ('alert', 'warn')
ORDER BY metric_time DESC;
```

Latest model refresh decisions:

```sql
SELECT *
FROM IDENTIFIER(:catalog || '.monitoring.model_refresh_decisions')
ORDER BY decision_time DESC
LIMIT 20;
```
