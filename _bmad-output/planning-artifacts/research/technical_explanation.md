# Technical Explanation: BTC Features And Models

## Feature Set

Features are generated from `btc_dev.raw.btc_hourly` in `notebooks/02_feature_engineering.py` and written to `btc_dev.features.btc_features`.

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
- Reads `btc_dev.features.btc_features`.
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

## Important Leakage Warning

The current feature table includes same-candle columns such as:

```text
open, high, low, close, volume, return_1h, hl_spread, oc_change
```

If the target is also the same row's `close`, the model may learn a leaky or trivial mapping rather than a real next-hour forecast.

For a true next-hour prediction task, the target is shifted forward:

```text
target_close_1h = lead(close, 1)
```

Then the model should train on current/past features to predict the next candle's close.

Current adjustment:
- `target_close_1h` is added in feature engineering.
- Training notebooks use `target_close_1h` as the target.
- Prediction output represents expected close for the next hour.

Remaining modeling caveat:
- Current-row OHLCV features are still used to predict the next-hour close.
- This is acceptable if prediction runs after the current candle has closed.
- If prediction must be made before candle close, remove current-row `high`, `low`, `close`, `volume`, `quote_volume`, `trades`, `return_1h`, `hl_spread`, and `oc_change` from the feature set.
