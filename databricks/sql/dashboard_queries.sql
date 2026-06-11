-- BTC MLOps AI/BI Dashboard Queries
-- Use these queries as dashboard datasets/tiles in Databricks SQL or AI/BI Dashboard.
-- Create a query/dashboard parameter named catalog, e.g. btc_simply or btc_prod.

-- 1. Data Freshness Tile
SELECT
  metric_time,
  metric_value AS raw_freshness_hours,
  status,
  details
FROM IDENTIFIER(:catalog || '.monitoring.pipeline_metrics')
WHERE metric_name = 'raw_freshness_hours'
ORDER BY metric_time DESC
LIMIT 1;

-- 2. Raw Data Count Tile
SELECT
  metric_time,
  metric_value AS raw_count,
  status
FROM IDENTIFIER(:catalog || '.monitoring.pipeline_metrics')
WHERE metric_name = 'raw_count'
ORDER BY metric_time DESC
LIMIT 1;

-- 3. Feature Table Count Tile
SELECT
  metric_time,
  metric_value AS features_count,
  status
FROM IDENTIFIER(:catalog || '.monitoring.pipeline_metrics')
WHERE metric_name = 'features_count'
ORDER BY metric_time DESC
LIMIT 1;

-- 4. Prediction Count Tile
SELECT
  metric_time,
  metric_value AS prediction_count,
  status
FROM IDENTIFIER(:catalog || '.monitoring.pipeline_metrics')
WHERE metric_name = 'prediction_count'
ORDER BY metric_time DESC
LIMIT 1;

-- 5. Latest Predictions Table
SELECT
  prediction_time,
  feature_open_time,
  predicted_close,
  predicted_return_1h,
  model_target_col,
  model_uri
FROM IDENTIFIER(:catalog || '.predictions.btc_predictions')
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
FROM IDENTIFIER(:catalog || '.predictions.btc_predictions') p
LEFT JOIN IDENTIFIER(:catalog || '.raw.btc_hourly') r
  ON r.open_time = p.feature_open_time + INTERVAL 1 HOUR
WHERE p.predicted_close > 1000.0
ORDER BY p.prediction_time DESC
LIMIT 100;

-- 7. Prediction Error Trend
SELECT
  p.feature_open_time,
  p.predicted_close,
  r.close AS actual_close,
  ABS(r.close - p.predicted_close) AS abs_error,
  ABS(r.close - p.predicted_close) / ABS(r.close) AS pct_error
FROM IDENTIFIER(:catalog || '.predictions.btc_predictions') p
JOIN IDENTIFIER(:catalog || '.raw.btc_hourly') r
  ON r.open_time = p.feature_open_time + INTERVAL 1 HOUR
WHERE p.predicted_close > 1000.0
ORDER BY p.feature_open_time;

-- 7b. Prediction Debug: find legacy return-as-close rows
SELECT
  prediction_time,
  feature_open_time,
  predicted_close,
  predicted_return_1h,
  model_target_col,
  model_version,
  model_run_id
FROM IDENTIFIER(:catalog || '.predictions.btc_predictions')
WHERE predicted_close <= 1000.0 OR predicted_close IS NULL
ORDER BY prediction_time DESC
LIMIT 100;

-- 8. Monitoring Alerts Table
SELECT
  metric_time,
  metric_name,
  metric_value,
  status,
  details
FROM IDENTIFIER(:catalog || '.monitoring.pipeline_metrics')
WHERE status IN ('alert', 'warn')
ORDER BY metric_time DESC
LIMIT 100;

-- 9. Monitoring Metrics Timeline
SELECT
  metric_time,
  metric_name,
  metric_value,
  status,
  details
FROM IDENTIFIER(:catalog || '.monitoring.pipeline_metrics')
ORDER BY metric_time DESC, metric_name
LIMIT 500;

-- 10. Latest Drift Metrics
SELECT
  metric_time,
  metric_name,
  metric_value,
  status,
  details
FROM IDENTIFIER(:catalog || '.monitoring.pipeline_metrics')
WHERE metric_name RLIKE '^(data_drift|label_drift|prediction_drift|model_drift|concept_drift|feature_quality|schema_drift|training_trigger)_'
ORDER BY metric_time DESC, metric_name
LIMIT 300;

-- 11. Drift Alerts And Warnings
SELECT
  metric_time,
  metric_name,
  metric_value,
  status,
  details
FROM IDENTIFIER(:catalog || '.monitoring.pipeline_metrics')
WHERE metric_name RLIKE '^(data_drift|label_drift|prediction_drift|model_drift|concept_drift|feature_quality|schema_drift|training_trigger)_'
  AND status IN ('alert', 'warn')
ORDER BY metric_time DESC, metric_name
LIMIT 100;

-- 12. BTC Trading Volume Trend
SELECT
  DATE_TRUNC('day', open_time) AS bucket_date,
  SUM(volume) AS btc_volume,
  SUM(quote_volume) AS quote_volume_usdt,
  COUNT(*) AS candle_count
FROM IDENTIFIER(:catalog || '.raw.btc_hourly')
WHERE open_time >= :date_range.min
  AND open_time <= :date_range.max
GROUP BY DATE_TRUNC('day', open_time)
ORDER BY bucket_date;

-- 13. Latest Model SHAP Explanation
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
  e.sample_rows,
  e.features_table_version,
  e.feature_config_id
FROM IDENTIFIER(:catalog || '.monitoring.model_explanations') e
JOIN latest_run r
  ON e.run_id = r.run_id
WHERE e.explanation_type = 'shap_summary'
ORDER BY e.mean_abs_shap DESC
LIMIT 30;

-- 14. Latest Model Built-In Feature Importance
WITH latest_run AS (
  SELECT run_id
  FROM IDENTIFIER(:catalog || '.monitoring.model_explanations')
  WHERE explanation_type = 'feature_importance'
  ORDER BY created_at DESC
  LIMIT 1
)
SELECT
  e.created_at,
  e.run_id,
  e.model_algo,
  e.feature,
  e.importance,
  e.features_table_version,
  e.feature_config_id
FROM IDENTIFIER(:catalog || '.monitoring.model_explanations') e
JOIN latest_run r
  ON e.run_id = r.run_id
WHERE e.explanation_type = 'feature_importance'
ORDER BY e.importance DESC
LIMIT 30;
