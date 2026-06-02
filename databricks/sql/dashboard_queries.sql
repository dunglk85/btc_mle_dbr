-- BTC MLOps AI/BI Dashboard Queries
-- Use these queries as dashboard datasets/tiles in Databricks SQL or AI/BI Dashboard.

-- 1. Data Freshness Tile
SELECT
  metric_time,
  metric_value AS raw_freshness_hours,
  status,
  details
FROM btc_dev.monitoring.pipeline_metrics
WHERE metric_name = 'raw_freshness_hours'
ORDER BY metric_time DESC
LIMIT 1;

-- 2. Raw Data Count Tile
SELECT
  metric_time,
  metric_value AS raw_count,
  status
FROM btc_dev.monitoring.pipeline_metrics
WHERE metric_name = 'raw_count'
ORDER BY metric_time DESC
LIMIT 1;

-- 3. Feature Table Count Tile
SELECT
  metric_time,
  metric_value AS features_count,
  status
FROM btc_dev.monitoring.pipeline_metrics
WHERE metric_name = 'features_count'
ORDER BY metric_time DESC
LIMIT 1;

-- 4. Prediction Count Tile
SELECT
  metric_time,
  metric_value AS prediction_count,
  status
FROM btc_dev.monitoring.pipeline_metrics
WHERE metric_name = 'prediction_count'
ORDER BY metric_time DESC
LIMIT 1;

-- 5. Latest Predictions Table
SELECT
  prediction_time,
  feature_open_time,
  predicted_close,
  model_uri
FROM btc_dev.predictions.btc_predictions
ORDER BY prediction_time DESC
LIMIT 50;

-- 6. Actual Vs Predicted Table
SELECT
  p.prediction_time,
  p.feature_open_time,
  p.predicted_close,
  r.open_time AS actual_open_time,
  r.close AS actual_close,
  ABS(r.close - p.predicted_close) AS abs_error,
  ABS(r.close - p.predicted_close) / ABS(r.close) AS pct_error
FROM btc_dev.predictions.btc_predictions p
LEFT JOIN btc_dev.raw.btc_hourly r
  ON r.open_time = p.feature_open_time + INTERVAL 1 HOUR
ORDER BY p.prediction_time DESC
LIMIT 100;

-- 7. Prediction Error Trend
SELECT
  p.feature_open_time,
  p.predicted_close,
  r.close AS actual_close,
  ABS(r.close - p.predicted_close) AS abs_error,
  ABS(r.close - p.predicted_close) / ABS(r.close) AS pct_error
FROM btc_dev.predictions.btc_predictions p
JOIN btc_dev.raw.btc_hourly r
  ON r.open_time = p.feature_open_time + INTERVAL 1 HOUR
ORDER BY p.feature_open_time;

-- 8. Model Refresh Decisions Table
SELECT
  decision_time,
  should_retrain,
  reason,
  trigger_mode,
  raw_freshness_hours,
  alert_count,
  champion_exists
FROM btc_dev.monitoring.model_refresh_decisions
ORDER BY decision_time DESC
LIMIT 50;

-- 9. Monitoring Alerts Table
SELECT
  metric_time,
  metric_name,
  metric_value,
  status,
  details
FROM btc_dev.monitoring.pipeline_metrics
WHERE status IN ('alert', 'warn')
ORDER BY metric_time DESC
LIMIT 100;

-- 10. Monitoring Metrics Timeline
SELECT
  metric_time,
  metric_name,
  metric_value,
  status,
  details
FROM btc_dev.monitoring.pipeline_metrics
ORDER BY metric_time DESC, metric_name
LIMIT 500;
