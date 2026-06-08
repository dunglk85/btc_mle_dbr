# BTC MLOps Monitoring Dashboard And Alerts

## Prerequisites

Run the data prediction job at least once for operational and prediction metrics:

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
- Training dataset manifests and replay validation status from recent model refresh runs.
- Prediction/model lineage table for `btc_predictions`.
- Latest model explanation: top SHAP features and built-in feature importance.
- Monitoring alerts table.
- Job quality metrics and alerts.

Drift tiles:
- Data drift PSI/KS by feature.
- Rolling RMSE/MAE/MAPE.
- Direction accuracy.
- Prediction error p95.
- Label drift for `target_close_1h`.
- Prediction drift for `predicted_close`.

Job quality tiles:
- Job success rate.
- Latest run duration.
- Failed run count.
- Failed task count.
- Job quality alerts and warnings.

Job quality alert trace query:

```sql
SELECT
  metric_time,
  metric_name,
  metric_value,
  status,
  from_json(details, 'job_id BIGINT, job_name STRING, job_url STRING, run_id BIGINT, run_url STRING, state STRUCT<life_cycle_state:STRING,result_state:STRING,state_message:STRING>, failed_tasks ARRAY<STRUCT<task_key:STRING,run_id:BIGINT,state:STRUCT<life_cycle_state:STRING,result_state:STRING,state_message:STRING>>>>') AS trace
FROM IDENTIFIER(:catalog || '.monitoring.pipeline_metrics')
WHERE metric_time >= current_timestamp() - INTERVAL 2 HOURS
  AND status = 'alert'
  AND metric_name RLIKE '^job_quality_'
ORDER BY metric_time DESC;
```

## SQL Alerts

SQL Alerts are managed by Databricks Asset Bundles in:

```text
databricks/resources/alerts.yml
```

Set `sql_warehouse_id` before deploying the bundle:

```bash
databricks bundle deploy --var="sql_warehouse_id=<warehouse-id>"
```

The SQL templates below remain useful for manual inspection or dashboard query creation.

Use the SQL queries in:

```text
databricks/sql/alert_queries.sql
```

For manual alert creation, create an alert query parameter named `catalog` and set it to `btc_dev` or `btc_prod`.

## Dashboard CI/CD

AI/BI Dashboard resources are supported by Databricks Asset Bundles, but they require a Lakeview dashboard JSON asset, usually exported as `.lvdash.json` from Databricks.

Current repository status:
- SQL alert resources are deployable by CI/CD.
- Dashboard SQL templates are versioned in `databricks/sql/dashboard_queries.sql`.
- Dashboard layout is stored in `databricks/dashboards/BTC MLOps Monitoring Dashboard.lvdash.json`.
- Dashboard resource is defined in `databricks/resources/dashboards.yml`.

Deploy dashboard and alerts:

```bash
databricks bundle deploy --var="sql_warehouse_id=632f18779c3b51ec"
```

Current SQL warehouse:

```text
632f18779c3b51ec  Serverless Starter Warehouse
```

Note: the exported dashboard currently contains `btc_dev` table references. It is suitable for dev deployment. For prod, export a prod dashboard or parameterize the dashboard JSON before deploying to `prod`.

Recommended alert conditions:
- Raw data stale: `raw_freshness_hours > 3`.
- Monitoring has alerts: `alert_count > 0`.
- No recent prediction: `prediction_age_hours > 3`.
- High prediction error: at least 12 evaluated predictions and `avg_pct_error > 0.05`.
- Feature target nulls beyond expected last row: `target_null_count > 1`.
- Job quality alert count: `job_quality_alert_count > 0`.

Drift alert conditions:
- Data drift PSI exceeds threshold, for example `psi > 0.2`.
- Rolling RMSE exceeds Champion validation RMSE by configured multiplier.
- Direction accuracy drops below configured threshold.

Drift metrics are written by:

```text
notebooks/08_drift_monitoring.py
```

Drift metrics are produced by `btc_drift_monitoring_job`, which runs separately from the hourly prediction job.

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

Latest training dataset manifests:

```sql
SELECT
  created_at,
  run_id,
  model_algo,
  target_col,
  raw_table_version,
  features_table_version,
  feature_config_version,
  feature_config_id,
  train_rows,
  test_rows,
  n_features
FROM IDENTIFIER(:catalog || '.monitoring.training_dataset_manifests')
ORDER BY created_at DESC
LIMIT 20;
```

Prediction lineage:

```sql
SELECT
  prediction_time,
  feature_open_time,
  predicted_close,
  predicted_return_1h,
  model_version,
  model_run_id,
  model_target_col,
  raw_table_version AS serving_raw_version,
  features_table_version AS serving_features_version,
  model_raw_table_version AS training_raw_version,
  model_features_table_version AS training_features_version,
  model_feature_config_version,
  model_feature_config_id
FROM IDENTIFIER(:catalog || '.predictions.btc_predictions')
ORDER BY prediction_time DESC
LIMIT 50;
```

Latest model SHAP explanation:

```sql
WITH latest_run AS (
  SELECT run_id
  FROM IDENTIFIER(:catalog || '.monitoring.model_explanations')
  WHERE explanation_type = 'shap_summary'
  ORDER BY created_at DESC
  LIMIT 1
)
SELECT
  e.created_at,
  e.run_id,
  e.model_algo,
  e.feature,
  e.mean_abs_shap,
  e.mean_shap,
  e.sample_rows
FROM IDENTIFIER(:catalog || '.monitoring.model_explanations') e
JOIN latest_run r ON e.run_id = r.run_id
WHERE e.explanation_type = 'shap_summary'
ORDER BY e.mean_abs_shap DESC
LIMIT 30;
```

Use this as a horizontal bar chart with `feature` on the axis and `mean_abs_shap` as the value.
