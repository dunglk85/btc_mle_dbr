-- BTC MLOps SQL Alert Queries
-- Create Databricks SQL Alerts from these queries if SQL Alerts are available.
-- Create a query/alert parameter named catalog, e.g. btc_dev or btc_prod.

-- Alert 1: Raw Data Stale
-- Condition: raw_freshness_hours > 3
SELECT
  metric_value AS raw_freshness_hours
FROM IDENTIFIER(:catalog || '.monitoring.pipeline_metrics')
WHERE metric_name = 'raw_freshness_hours'
ORDER BY metric_time DESC
LIMIT 1;

-- Alert 2: Monitoring Has Alerts
-- Condition: alert_count > 0
SELECT COUNT(*) AS alert_count
FROM IDENTIFIER(:catalog || '.monitoring.pipeline_metrics')
WHERE metric_time >= current_timestamp() - INTERVAL 2 HOURS
  AND status = 'alert';

-- Alert 3: No Recent Prediction
-- Condition: prediction_age_hours > 3
SELECT
  CASE
    WHEN MAX(prediction_time) IS NULL THEN 999999
    ELSE TIMESTAMPDIFF(
      HOUR,
      MAX(prediction_time),
      current_timestamp()
    )
  END AS prediction_age_hours
FROM IDENTIFIER(:catalog || '.predictions.btc_predictions');

-- Alert 4: High Prediction Error
-- Condition: avg_pct_error > 0.02
SELECT
  AVG(ABS(r.close - p.predicted_close) / ABS(r.close)) AS avg_pct_error
FROM IDENTIFIER(:catalog || '.predictions.btc_predictions') p
JOIN IDENTIFIER(:catalog || '.raw.btc_hourly') r
  ON r.open_time = p.feature_open_time + INTERVAL 1 HOUR
WHERE p.prediction_time >= current_timestamp() - INTERVAL 24 HOURS;

-- Alert 5: Feature Table Missing Target Values Beyond Expected Last Row
-- Condition: target_null_count > 1
SELECT
  COUNT(*) AS target_null_count
FROM IDENTIFIER(:catalog || '.features.btc_features')
WHERE target_close_1h IS NULL;
