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
- High prediction error: `avg_pct_error > 0.02`.
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

The data prediction job runs drift monitoring after regular monitoring.

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
