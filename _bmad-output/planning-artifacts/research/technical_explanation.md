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

## Monitoring Gate Metrics

The model refresh gate is implemented in:

```text
notebooks/07_monitoring_gate.py
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
- If `alert_count > 0`, the gate currently skips retraining.
- This avoids training on potentially bad or stale data.

### champion_exists

Meaning:
- Whether `<catalog>.models.btc_price_model@Champion` exists.

Usage:
- If no Champion exists and monitoring is healthy, retraining is allowed so the project can create the first Champion.

## Data Drift And Model Drift

### Data Drift

Data drift means the input data distribution changes compared with the data used to train the Champion model.

For this project, examples include:
- BTC hourly volume distribution changes sharply.
- Return distribution becomes more volatile than the training window.
- Feature ranges shift, such as `close`, `volume`, `quote_volume`, `return_1h`, or moving averages.
- Missing hourly candles become more frequent.

Recommended data drift metrics:

```text
feature_mean_delta
feature_std_delta
feature_null_rate_delta
feature_quantile_delta
population_stability_index
ks_statistic
missing_hour_count
```

Suggested baseline:
- Use the training dataset statistics logged with the Champion model.
- Compare recent production feature windows, such as last 24h / 7d, against that baseline.

Current implementation status:
- Implemented in `notebooks/08_drift_monitoring.py` for selected features using PSI and approximate KS statistics.
- Metrics are written to `<catalog>.monitoring.pipeline_metrics` with names like `data_drift_psi_close` and `data_drift_ks_return_1h`.

### Model Drift / Performance Drift

Model drift means the Champion model becomes less accurate because the relationship between features and target changes over time.

For this project, performance drift can be measured once actual next-hour closes are available.

Prediction-vs-actual join:

```sql
predictions.feature_open_time + INTERVAL 1 HOUR = raw.open_time
```

Recommended model drift metrics:

```text
rolling_rmse_24h
rolling_mae_24h
rolling_mape_24h
rolling_direction_accuracy_24h
error_quantile_p95
champion_vs_challenger_rmse_delta
```

Suggested alert conditions:
- Rolling RMSE exceeds Champion validation RMSE by a configured multiplier.
- Rolling MAPE exceeds a fixed threshold.
- Direction accuracy drops below a minimum threshold.
- Prediction error p95 spikes sharply.

Current implementation status:
- Actual-vs-predicted SQL is available in `databricks/sql/dashboard_queries.sql`.
- High average prediction error SQL alert is available in `databricks/sql/alert_queries.sql`.
- Rolling performance metrics are persisted by `notebooks/08_drift_monitoring.py`.

### Recommended Next Implementation

Dedicated notebook:

```text
notebooks/08_drift_monitoring.py
```

It writes drift metrics to:

```text
<catalog>.monitoring.pipeline_metrics
```

Recommended metric names:

```text
data_drift_psi_close
data_drift_psi_volume
data_drift_ks_return_1h
model_drift_rmse_24h
model_drift_mape_24h
model_drift_direction_accuracy_24h
```

The monitoring gate uses these metrics to set `should_retrain = true` when drift is detected and data quality is otherwise healthy. Blocking schema/quality alerts still stop retraining.

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
