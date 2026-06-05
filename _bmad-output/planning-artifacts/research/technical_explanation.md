# Technical Explanation: BTC Features And Models

## Feature Set

Features are generated from `<catalog>.raw.btc_hourly` in `notebooks/02_feature_engineering.py` and written to `<catalog>.features.btc_features`. Databricks jobs pass the `catalog` widget, defaulting to `btc_dev` in development and `btc_prod` in production.

### OHLCV Base Columns

Columns:

```text
open, high, low, close, volume
```

Meaning:
- `open`: candle opening price.
- `high`: highest price in the hour.
- `low`: lowest price in the hour.
- `close`: candle closing price.
- `volume`: BTC trading volume.

Purpose:
- These are the core market price and volume signals.
- They describe price movement and trading activity during each hourly candle.

### Quote Volume And Trades

Columns:

```text
quote_volume, trades
```

Meaning:
- `quote_volume`: volume measured in the quote asset, normally USDT.
- `trades`: number of trades in the candle.

Purpose:
- These features indicate market activity and liquidity.
- Higher trade count or quote volume can be associated with stronger price movement.

### Moving Averages

Columns:

```text
ma_7, ma_24, ma_168
```

Meaning:
- `ma_7`: average close price over the previous 7 hours.
- `ma_24`: average close price over the previous 24 hours.
- `ma_168`: average close price over the previous 168 hours, approximately 7 days.

Purpose:
- `ma_7` captures short-term trend.
- `ma_24` captures daily trend.
- `ma_168` captures weekly trend.
- The model can compare recent price action against short, daily, and weekly context.

### Lag Features

Columns:

```text
close_lag_1h, close_lag_2h, close_lag_4h, close_lag_12h, close_lag_24h
```

Meaning:
- These are previous close prices from 1, 2, 4, 12, and 24 hours ago.

Purpose:
- Time series data often depends heavily on recent past values.
- Lag features help the model learn momentum, reversal, and local price structure.

### One-Hour Return

Column:

```text
return_1h = close / close_lag_1h - 1
```

Meaning:
- Percentage price change over the last hour.

Purpose:
- Captures short-term momentum.
- Positive return means recent upward movement; negative return means recent downward movement.

### High-Low Spread

Column:

```text
hl_spread = high - low
```

Meaning:
- Price range inside the hourly candle.

Purpose:
- Acts as a simple volatility signal.
- Wider spread indicates stronger intrahour movement.

### Open-Close Change

Column:

```text
oc_change = close - open
```

Meaning:
- Difference between candle close and candle open.

Purpose:
- Shows whether the candle closed higher or lower than it opened.
- Useful as a simple buying/selling pressure signal.

### Time Features

Columns:

```text
hour, day_of_week
```

Meaning:
- `hour`: hour of day.
- `day_of_week`: day of week.

Purpose:
- Crypto trades 24/7, but volume and volatility can still vary by hour and day.
- These features allow the model to learn weak seasonality patterns.

## Models

### Baseline Training Notebook

Notebook:

```text
notebooks/03_model_training.py
```

Model:

```text
RandomForestRegressor
```

Default configuration:

```text
n_estimators = 100
max_depth = 10
random_state = 42
```

Training process:
- Reads `<catalog>.features.btc_features`.
- Drops rows with null values caused by lag and moving-average features.
- Uses temporal split, not random split.
- First 80% of time-ordered data is used for training.
- Last 20% is used for evaluation.

Metrics:

```text
rmse, mae, r2, mape
```

Why RandomForest first:
- Simple and stable baseline.
- Works well enough for initial pipeline validation.
- Does not require GPU.
- Lower operational risk on Databricks Free Edition.
- Easier to debug than more complex boosted-tree workflows.

### Optuna Training Notebook

Notebook:

```text
notebooks/03_optuna_training.py
```

Model:

```text
RandomForestRegressor
```

Optuna tunes:

```text
n_estimators
max_depth
min_samples_split
min_samples_leaf
```

Default optimization settings:

```text
n_trials = 15
```

Purpose:
- Search for a better RandomForest configuration.
- Minimize RMSE.
- Log each trial to MLflow.
- Log the best model as an MLflow model artifact.

## Metrics

The training notebooks log four core metrics:

```text
rmse, mae, r2, mape
```

These metrics evaluate prediction quality on the time-based holdout set.

### RMSE

Full name:

```text
Root Mean Squared Error
```

Conceptual formula:

```text
sqrt(mean((actual - predicted)^2))
```

Meaning:
- Measures average prediction error with stronger penalty for large mistakes.
- Unit is the same as the target, so here it is BTC price in USD.
- Example: `RMSE = 500` means the model has material prediction error around hundreds of USD, with large misses weighted heavily.

Usage:
- Current primary metric for Champion/Challenger promotion.
- Lower is better.
- Useful when large forecast errors are especially undesirable.

### MAE

Full name:

```text
Mean Absolute Error
```

Conceptual formula:

```text
mean(abs(actual - predicted))
```

Meaning:
- Measures average absolute prediction error.
- Unit is also USD.
- Example: `MAE = 300` means predictions are off by about 300 USD on average.

Usage:
- Easier to interpret than RMSE.
- Lower is better.
- Less sensitive to outliers than RMSE.

### R2

Full name:

```text
Coefficient of Determination
```

Meaning:
- Measures how much target variance is explained by the model.
- `1.0` means perfect prediction.
- `0.0` means not better than predicting the average.
- Negative values mean the model is worse than a naive average baseline.

Usage:
- Higher is better.
- Useful as a relative signal that the model is learning structure.
- For financial time series, high R2 does not automatically mean the model will generalize well, because market behavior can shift over time.

### MAPE

Full name:

```text
Mean Absolute Percentage Error
```

Conceptual formula:

```text
mean(abs(actual - predicted) / abs(actual))
```

Meaning:
- Measures prediction error as a percentage of the actual price.
- Example: `MAPE = 0.005` means average error is about `0.5%`.

Usage:
- Lower is better.
- Easier to compare across price regimes than raw USD error.
- Good for dashboards because it is interpretable as a percentage.

### Reading Metrics Together

Do not judge the model using only one metric.

Interpretation examples:
- High RMSE but low MAE means most predictions are acceptable, but a few large misses exist.
- High RMSE and high MAE means the model is broadly inaccurate.
- Low MAPE but high RMSE means percentage error may look acceptable while USD error is still large.
- Low R2 with acceptable MAE/MAPE can still be usable for an MLOps demo, but it suggests weak explanatory power.

### Champion Promotion Metric

`notebooks/04_champion_challenger.py` currently promotes a Challenger when its RMSE is lower than the current Champion's RMSE.

Rationale:
- RMSE penalizes large misses more heavily than MAE.
- For BTC price forecasting, avoiding large forecast errors is important.
- The first valid Challenger is promoted if no Champion exists.

### Trading Metrics Caveat

These metrics evaluate price forecast accuracy, not trading profitability.

A model with good RMSE/MAE/MAPE may still be poor for trading because these metrics do not account for:
- Trading fees.
- Slippage.
- Direction accuracy.
- Drawdown.
- Position sizing.
- Risk-adjusted returns.

If the project evolves into a trading strategy, add metrics such as:

```text
direction_accuracy
hit_rate
profit_factor
max_drawdown
sharpe_ratio
```

## Monitoring Metrics

Monitoring metrics are written by:

```text
notebooks/06_monitoring.py
```

Current monitoring is an operational fallback based on Delta tables. It checks pipeline health, freshness, row counts, duplicates, target nulls, prediction availability, and prediction age. It does not yet implement full statistical drift detection.

to:

```text
<catalog>.monitoring.pipeline_metrics
```

Each metric row has:

```text
metric_time
metric_name
metric_value
status
```

Status values:
- `ok`: metric is healthy.
- `warn`: non-critical issue or missing optional object.
- `alert`: critical issue; monitoring notebook raises an error so the Databricks Job can notify operators.

### raw_count

Meaning:
- Number of rows in `<catalog>.raw.btc_hourly`.

Purpose:
- Confirms raw ingestion is producing data.
- Used as a basic pipeline health check.

Expected behavior:
- Should be greater than zero.
- Should grow over time as new hourly candles arrive.

### raw_duplicate_open_time_count

Meaning:
- Count of duplicated `open_time` values in the raw table.

Purpose:
- Verifies ingestion idempotency.
- Since raw ingestion merges on `open_time`, duplicates should not exist.

Healthy threshold:

```text
raw_duplicate_open_time_count = 0
```

Alert condition:

```text
raw_duplicate_open_time_count > 0
```

### raw_null_open_time_count

Meaning:
- Count of rows where `open_time` is null in the raw table.

Purpose:
- Detects malformed candle data or timestamp parsing failures.
- `open_time` is the natural key for merge and ordering, so null values are critical.

Healthy threshold:

```text
raw_null_open_time_count = 0
```

Alert condition:

```text
raw_null_open_time_count > 0
```

### raw_freshness_hours

Meaning:
- Age in hours between current time and latest raw candle `open_time`.

Purpose:
- Detects stale ingestion.
- Ensures the pipeline is receiving recent market data.

Current threshold:

```text
raw_freshness_hours <= 3
```

Alert condition:

```text
raw_freshness_hours > 3
```

Why 3 hours:
- Data is hourly.
- A small delay is acceptable, but more than 3 hours suggests fetch, upload, or ingestion is broken.

### features_count

Meaning:
- Number of rows in `<catalog>.features.btc_features`.

Purpose:
- Confirms feature engineering created the feature table.
- Used to compare raw rows vs feature rows.

Expected behavior:
- Should be close to `raw_count`.
- A small difference can occur depending on target/lag handling.

### features_target_close_1h_null_count

Meaning:
- Number of rows where `target_close_1h` is null.

Purpose:
- Validates next-hour target generation.
- `target_close_1h` is populated only when an exact candle exists at `open_time + 1 hour`.
- Rows without an exact next-hour candle have null target values.

Healthy threshold:

```text
features_target_close_1h_null_count <= 1 under normal no-gap data
```

Alert condition:

```text
features_target_close_1h_null_count > 1
```

### raw_features_row_count_delta

Meaning:
- Difference between raw row count and feature row count.

Formula:

```text
raw_count - features_count
```

Purpose:
- Detects whether feature engineering lost rows unexpectedly.

Healthy threshold:

```text
abs(raw_features_row_count_delta) <= 1
```

Alert condition:

```text
abs(raw_features_row_count_delta) > 1
```

### prediction_count

Meaning:
- Number of rows in `<catalog>.predictions.btc_predictions`.

Purpose:
- Confirms prediction output exists.
- If no Champion model exists yet, prediction may be skipped, so this can be `warn` early in the project.

Expected behavior:
- Should increase over time after a Champion model is available.
- Should not duplicate endlessly because prediction writes use merge semantics by `feature_open_time` and `model_uri`.

### prediction_age_hours

Meaning:
- Age in hours since the latest prediction was written.

Purpose:
- Detects stale prediction output.
- Ensures the serving path is producing predictions after new data arrives.

Current threshold:

```text
prediction_age_hours <= 3
```

Alert condition:

```text
prediction_age_hours > 3
```

## Training Gate Metrics

The model refresh gate is implemented in:

```text
notebooks/07_training_gate.py
```

It writes decisions to:

```text
<catalog>.monitoring.model_refresh_decisions
```

Decision columns:

```text
should_retrain
reason
trigger_mode
raw_freshness_hours
alert_count
champion_exists
```

### should_retrain

Meaning:
- Boolean decision telling the model refresh job whether to train a new model.

Usage:
- `03_optuna_training.py` reads the latest decision.
- If `should_retrain = false`, training exits with `SKIP_RETRAIN`.
- If `should_retrain = true`, Optuna training proceeds.

### reason

Meaning:
- Human-readable explanation for the decision.

Examples:
- `no Champion exists`
- `trigger_mode=manual`
- `latest monitoring has 2 alert metrics`
- `raw data stale: 5.20h > 3.00h`

Purpose:
- Makes gate behavior auditable.
- Helps explain why retraining did or did not happen.

### trigger_mode

Meaning:
- Source or intent of the model refresh run.

Current values:
- `scheduled`: periodic refresh.
- `manual`: operator-triggered refresh.
- `drift`: future monitoring-triggered refresh.

Purpose:
- Allows different retraining policies depending on why the job was triggered.

### alert_count

Meaning:
- Number of alert metrics in the latest monitoring snapshot.

Usage:
- Blocking quality alerts skip retraining.
- Model/concept drift alerts can trigger retraining when data/schema/feature quality validation passes.

### champion_exists

Meaning:
- Whether `<catalog>.models.btc_price_model@Champion` exists.

Usage:
- If no Champion exists and monitoring is healthy, retraining is allowed so the project can create the first Champion.

## Drift Metrics

Drift metrics are written by:

```text
notebooks/08_drift_monitoring.py
```

to:

```text
<catalog>.monitoring.pipeline_metrics
```

The notebook compares two time windows:
- **Recent window:** latest `recent_hours`, default `168` hours.
- **Reference window:** previous `reference_hours`, default `720` hours.

This makes drift monitoring self-contained and independent of Databricks Lakehouse Monitoring. It is a practical fallback for Databricks Free Edition/serverless.

### Data Drift Metrics

Data drift means the input feature distribution changed compared with the reference window.

Current feature columns monitored:

```text
volume
quote_volume
trades
return_1h
```

Price-level and price-derived columns are tracked separately as monitor-only drift metrics:

```text
close
ma_24
target_close_1h
predicted_close
```

Reason:
- BTC price-level distributions naturally shift when the market trends.
- `ma_24` is derived from price level and can create the same false positives during trends.
- Comparing short recent windows against prior windows can produce constant PSI/KS alerts.
- These metrics are useful on the dashboard, but should not trigger retraining by themselves.

Metrics:

```text
data_drift_psi_<feature>
data_drift_ks_<feature>
```

Examples:

```text
data_drift_ks_return_1h
data_drift_psi_volume
```

#### PSI

PSI means Population Stability Index.

How it is calculated:
- Build quantile buckets from the reference window.
- Count the percentage of reference rows in each bucket.
- Count the percentage of recent rows in each bucket.
- Sum the bucket-level distribution difference.

Interpretation:
- `PSI < 0.25`: operationally stable for this BTC hourly pipeline.
- `0.25 <= PSI < 1.0`: warning drift.
- `PSI >= 1.0`: alert drift.

Current thresholds:

```text
psi_warn_threshold = 0.25
psi_alert_threshold = 1.0
```

Why useful:
- Captures distribution shift across the whole feature, not only mean/std.
- Good for detecting regime changes in BTC volume, returns, or price range.

#### Approximate KS

KS means Kolmogorov-Smirnov statistic.

How it is calculated here:
- Compute reference quantiles at 10%, 20%, ..., 90%.
- Estimate CDF difference between reference and recent windows at those quantiles.
- Use the maximum CDF difference as approximate KS.

Interpretation:
- Higher value means stronger distribution difference.
- The implementation is approximate because it avoids expensive full empirical CDF computation.

Current thresholds:

```text
ks_warn_threshold = 0.30
ks_alert_threshold = 0.60
```

### Label Drift Metrics

Label drift means the target distribution changed.

Metrics:

```text
label_drift_psi_target_close_1h
label_drift_ks_target_close_1h
```

Meaning:
- These are the same PSI/KS checks applied specifically to `target_close_1h`.
- They detect whether actual next-hour close values are moving into a different regime.

Important caveat:
- `target_close_1h` is null when the exact next-hour candle is missing.
- Rows with null target are excluded from drift distribution calculations.

Current behavior:
- Label drift is monitor-only.
- It can produce `warn`, but does not produce retraining `alert` under the current thresholds.
- This avoids retraining every time BTC price level naturally trends upward or downward.

### Prediction Drift Metrics

Prediction drift means the model output distribution changed, even before accuracy is evaluated.

Metrics:

```text
prediction_drift_psi_predicted_close
prediction_drift_ks_predicted_close
```

Meaning:
- Compares recent `predicted_close` distribution against previous prediction distribution.
- Useful when predictions start moving to unusual ranges.

Why separate from model drift:
- Prediction drift can happen even before actual labels are available.
- Model drift requires actual next-hour close to compute error.

Current behavior:
- Prediction drift is monitor-only.
- It can produce `warn`, but does not produce retraining `alert` under the current thresholds.
- Retraining should be triggered by performance degradation or non-price feature drift, not merely by BTC price level moving.

### Model / Performance Drift Metrics

Model drift means model quality degraded over time.

Actual-vs-predicted join:

```sql
predictions.feature_open_time + INTERVAL 1 HOUR = raw.open_time
```

Metrics:

```text
model_drift_joined_prediction_count
model_drift_rmse_24h
model_drift_mae_24h
model_drift_mape_24h
model_drift_p95_abs_error_24h
model_drift_direction_accuracy_24h
```

#### model_drift_joined_prediction_count

Meaning:
- Number of predictions that can be joined with actual next-hour close.

Why useful:
- If this is `0`, actuals are not available yet or prediction/raw timestamps are misaligned.
- Other performance drift metrics are unreliable when join count is too low.

#### model_drift_rmse_24h

Meaning:
- Root Mean Squared Error over recent predictions with actuals.

Formula:

```text
sqrt(avg((actual_close - predicted_close)^2))
```

Why useful:
- Penalizes large BTC forecast misses heavily.
- Good primary performance drift metric for regression.

#### model_drift_mae_24h

Meaning:
- Mean Absolute Error over recent predictions with actuals.

Formula:

```text
avg(abs(actual_close - predicted_close))
```

Why useful:
- Easier to interpret than RMSE because it is average absolute dollars of error.

#### model_drift_mape_24h

Meaning:
- Mean Absolute Percentage Error.

Formula:

```text
avg(abs(actual_close - predicted_close) / abs(actual_close))
```

Current thresholds:

```text
mape_warn_threshold = 0.02
mape_alert_threshold = 0.05
```

Interpretation:
- `MAPE >= 2%`: warning.
- `MAPE >= 5%`: alert.

#### model_drift_p95_abs_error_24h

Meaning:
- Approximate 95th percentile of absolute error.

Why useful:
- Captures tail-risk forecast misses.
- A model can have acceptable average error but still produce dangerous outlier misses.

#### model_drift_direction_accuracy_24h

Meaning:
- How often predicted direction matches actual direction.

Approximation:
- Actual direction: sign of change in actual close between consecutive actuals.
- Predicted direction: sign of change in predicted close between consecutive predictions.

Current thresholds:

```text
direction_warn_threshold = 0.45
direction_alert_threshold = 0.40
```

Interpretation:
- `<= 45%`: warning.
- `<= 40%`: alert.

Trading caveat:
- Direction accuracy alone is not a trading strategy metric.
- It does not include fees, slippage, sizing, or drawdown.

### Concept Drift Proxy Metric

Concept drift means the relationship between features and target changed.

Metric:

```text
concept_drift_mean_error_bias_24h
```

Meaning:
- Average signed error over recent predictions.

Formula:

```text
avg(actual_close - predicted_close)
```

Interpretation:
- Positive persistent bias means the model tends to underpredict.
- Negative persistent bias means the model tends to overpredict.
- This is a proxy metric, not a full causal concept drift test.

Why useful:
- A stable signed bias can indicate that the model relationship learned during training no longer matches the current market regime.

### Feature Quality Metrics

Feature quality drift means feature data quality degraded, regardless of whether the market distribution changed naturally.

Metrics:

```text
feature_quality_null_rate_<feature>
schema_drift_missing_<feature>
```

Examples:

```text
feature_quality_null_rate_close
feature_quality_null_rate_target_close_1h
schema_drift_missing_return_1h
```

#### feature_quality_null_rate_<feature>

Meaning:
- Percentage of recent rows where the feature is null.

Current thresholds:

```text
null_rate_warn_threshold = 0.05
null_rate_alert_threshold = 0.20
```

Interpretation:
- `>= 5%`: warning.
- `>= 20%`: alert.

Why useful:
- Stops retraining on broken feature tables.
- Detects upstream ingestion or feature engineering issues.

#### schema_drift_missing_<feature>

Meaning:
- Expected feature column is missing from the feature table.

Status:
- Always `alert` when emitted.

Why useful:
- Missing columns will break training or silently change model inputs.

### Drift Gate Behavior

The training gate treats model/concept drift alerts as retraining candidates, not automatic retraining approval.

Retraining decision flow:

```text
Model drift / concept drift alert
        ↓
Validate data quality + schema quality + feature quality
        ↓
If validation passes: should_retrain = true
If validation fails: should_retrain = false
```

Blocking alert types:

```text
raw_freshness_hours
raw_duplicate_open_time_count
raw_null_open_time_count
features_target_close_1h_null_count
raw_features_row_count_delta
feature_quality_*
schema_drift_*
```

Retraining trigger alert types:

```text
model_drift_*
concept_drift_*
```

Monitor-only drift types:

```text
price_level_drift_*
data_drift_*
label_drift_*
prediction_drift_*
```

These are shown in the dashboard but should not trigger retraining by themselves.

Immediate drift-triggered retraining is wired into `btc_data_prediction_job`:
- `drift_monitoring`
- `training_gate_drift` with `trigger_mode=drift`
- `model_training_drift`
- `champion_challenger_drift`

If no drift alert exists, `training_gate_drift` records `should_retrain=false` and training exits with `SKIP_RETRAIN`.

## Job Quality Metrics

Job quality metrics are written by:

```text
notebooks/09_job_quality_monitoring.py
```

to:

```text
<catalog>.monitoring.pipeline_metrics
```

These metrics monitor Databricks Jobs health using the Databricks Jobs API. They are shown in the dashboard page:

```text
Job Quality Monitoring
```

The notebook looks for jobs whose name contains the configured `job_name_filter`, default:

```text
BTC
```

It inspects the latest `lookback_runs`, default:

```text
20
```

### job_quality_matching_job_count

Meaning:
- Number of Databricks Jobs whose name matches `job_name_filter`.

Purpose:
- Confirms the monitoring notebook can discover the target jobs.
- Helps catch naming changes or permission/API visibility issues.

Expected behavior:
- Should be greater than zero.
- In this project, it should normally find the data prediction job and model refresh job.

### job_quality_success_rate_<job_id>

Meaning:
- Percentage of recent terminal runs for a job that ended in `SUCCESS`.

Formula:

```text
successful_terminal_runs / terminal_runs
```

Purpose:
- Measures reliability of each Databricks Job over recent runs.
- Detects repeated job failures even if the latest run happens to succeed.

Current threshold:

```text
min_success_rate = 0.8
```

Status behavior:
- `ok` when success rate is at least `0.8`.
- `alert` when success rate is below `0.8`.
- `warn` when there are no terminal runs to evaluate.

Interpretation:
- `1.0`: all recent terminal runs succeeded.
- `0.5`: only half of recent terminal runs succeeded.
- `null`: no terminal run exists in the lookback window.

### job_quality_failed_run_count_<job_id>

Meaning:
- Number of recent terminal runs for a job where `result_state != SUCCESS`.

Purpose:
- Gives an absolute count of failures in the lookback window.
- Easier to inspect than success rate when debugging incidents.

Status behavior:
- `ok` when failed run count is `0`.
- `alert` when failed run count is greater than `0`.

Interpretation:
- A non-zero value means at least one recent run failed and should be inspected in Databricks Jobs UI.

### job_quality_latest_duration_minutes_<job_id>

Meaning:
- Duration in minutes of the latest terminal run for a job.

Purpose:
- Detects jobs that are still succeeding but taking too long.
- Useful for catching slow API calls, slow Delta scans, slow Optuna training, or serverless cold-start/resource issues.

Current threshold:

```text
max_duration_minutes = 60
```

Status behavior:
- `ok` when latest duration is at most `60` minutes.
- `alert` when latest duration is greater than `60` minutes.
- `warn` when duration is missing.

Interpretation:
- For the hourly data prediction job, duration should be comfortably below one hour.
- If duration approaches the hourly schedule interval, runs can overlap or create delayed monitoring/prediction output.

### job_quality_latest_success_<job_id>

Meaning:
- Whether the latest terminal run succeeded.

Values:

```text
1 = latest terminal run succeeded
0 = latest terminal run failed
null = latest run is not terminal yet
```

Purpose:
- Fast, simple latest-run health signal.
- Complements success rate, which summarizes multiple runs.

Status behavior:
- `ok` when value is `1`.
- `alert` when value is `0` and the run is terminal.
- `warn` when the latest observed run is not terminal yet.

Why latest terminal run is used:
- The monitoring notebook can run while the current Databricks Job is still active.
- Treating the currently running job as failed would create false alerts.
- Therefore the notebook prefers the latest terminal run when available.

### job_quality_latest_failed_task_count_<job_id>

Meaning:
- Number of failed tasks in the latest terminal run.

Purpose:
- Identifies whether a job-level failure was caused by one or more failed tasks.
- Helps locate the broken stage in multi-task workflows.

Status behavior:
- `ok` when failed task count is `0`.
- `alert` when failed task count is greater than `0`.

Examples of tasks that can fail:
- `fetch_binance`
- `data_ingestion`
- `feature_engineering`
- `prediction`
- `monitoring`
- `drift_monitoring`
- `model_training_drift`
- `champion_challenger_drift`
- `job_quality_monitoring`

### Job Quality Alert

The SQL alert resource is defined in:

```text
databricks/resources/alerts.yml
```

Alert query:

```text
job_quality_alert
```

Alert condition:

```text
job_quality_alert_count > 0
```

It triggers when any recent metric with prefix `job_quality_` has status `alert`.

### How To Use The Dashboard Page

The dashboard page:

```text
Job Quality Monitoring
```

contains two tables:
- `Job Quality Alerts and Warnings`: only metrics with `alert` or `warn`.
- `Latest Job Quality Metrics`: all recent job quality metrics.

Recommended triage order:
- Check `job_quality_latest_success_<job_id>` to see whether the latest terminal run failed.
- Check `job_quality_latest_failed_task_count_<job_id>` to see if task-level failures exist.
- Check `job_quality_failed_run_count_<job_id>` and `job_quality_success_rate_<job_id>` to understand whether this is isolated or recurring.
- Check `job_quality_latest_duration_minutes_<job_id>` to detect slow jobs.

### Operational Notes

Job quality monitoring is not a model retraining trigger by itself.

Reason:
- Job quality alerts indicate orchestration or runtime reliability issues.
- Retraining on failure conditions may make the system worse if upstream tasks are broken.

Job quality alerts should trigger operator investigation, not automatic retraining.

## Dashboard And Alert Artifacts

Dashboard queries are stored in:

```text
databricks/sql/dashboard_queries.sql
```

SQL alert queries are stored in:

```text
databricks/sql/alert_queries.sql
```

Operational guide:

```text
docs/monitoring-dashboard.md
```

## Important Leakage Warning

The current feature table includes same-candle columns such as:

```text
open, high, low, close, volume, return_1h, hl_spread, oc_change
```

If the target is also the same row's `close`, the model may learn a leaky or trivial mapping rather than a real next-hour forecast.

For a true next-hour prediction task, the target is shifted forward:

```text
target_close_1h = close where target.open_time = current.open_time + 1 hour
```

Then the model should train on current/past features to predict the next candle's close.

Current adjustment:
- `target_close_1h` is added in feature engineering.
- Target generation uses an exact next-hour self-join rather than next-row `lead`, so missing hourly candles do not silently create mislabeled targets.
- Training notebooks use `target_close_1h` as the target.
- Prediction output represents expected close for the next hour.

Remaining modeling caveat:
- Current-row OHLCV features are still used to predict the next-hour close.
- This is acceptable if prediction runs after the current candle has closed.
- If prediction must be made before candle close, remove current-row `high`, `low`, `close`, `volume`, `quote_volume`, `trades`, `return_1h`, `hl_spread`, and `oc_change` from the feature set.
