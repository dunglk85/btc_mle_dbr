# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 02 - Feature Engineering (Advanced)
# MAGIC
# MAGIC Tạo candidate features cho bài toán regression.
# MAGIC
# MAGIC **Features bao gồm:**
# MAGIC - Return features (1h, 6h, 24h)
# MAGIC - Moving Average Ratios (MA7, MA24, MA168)
# MAGIC - Technical Indicators: MACD, RSI, ATR, Bollinger Bands
# MAGIC - Volume Ratios & Log Volume
# MAGIC - Cyclical Time Features (sin/cos cho giờ và thứ)
# MAGIC - Lag features
# MAGIC
# MAGIC **Targets:**
# MAGIC - `target_return_1h` (Regression): % thay đổi giá close giờ tiếp theo

# COMMAND ----------

# MAGIC %pip install lightgbm scikit-learn

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import json
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.feature_selection import mutual_info_regression
from pyspark.sql import Window, functions as F

# COMMAND ----------

def get_widget(name, default):
    try:
        dbutils.widgets.text(name, str(default))
        return dbutils.widgets.get(name)
    except Exception:
        return str(default)


catalog = get_widget("catalog", "btc_simply")
raw_schema = "raw"
features_schema = "features"
raw_table = "btc_hourly"
features_table = "btc_features"
raw_ref = f"{catalog}.{raw_schema}.{raw_table}"
features_ref = f"{catalog}.{features_schema}.{features_table}"
feature_config_ref = f"{catalog}.{features_schema}.feature_selection_config"
corr_threshold = float(get_widget("corr_threshold", "0.90"))
mi_threshold = float(get_widget("mi_threshold", "0.001"))

print("RUNNING ADVANCED FEATURE ENGINEERING NOTEBOOK")
print(f"raw_ref={raw_ref}")
print(f"features_ref={features_ref}")
print(f"feature_config_ref={feature_config_ref}")
print(f"corr_threshold={corr_threshold}")
print(f"mi_threshold={mi_threshold}")

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{features_schema}")

# COMMAND ----------

raw = spark.table(raw_ref).orderBy("open_time")
raw_count = raw.count()
print(f"raw_count={raw_count}")
if raw_count == 0:
    raise ValueError(f"No raw rows found in {raw_ref}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Base Window & Lag Features

# COMMAND ----------

w = Window.orderBy("open_time")
features = raw

# --- Lag features ---
for lag_hour in [1, 2, 4, 12, 24]:
    features = features.withColumn(
        f"close_lag_{lag_hour}h",
        F.lag("close", lag_hour).over(w),
    )

# --- Basic spread features ---
features = features.withColumn("hl_spread", F.col("high") - F.col("low"))
features = features.withColumn("oc_change", F.col("close") - F.col("open"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Return Features

# COMMAND ----------

features = features.withColumn(
    "return_1h", (F.col("close") / F.col("close_lag_1h")) - F.lit(1.0)
)
features = features.withColumn(
    "return_6h", (F.col("close") / F.lag("close", 6).over(w)) - F.lit(1.0)
)
features = features.withColumn(
    "return_24h", (F.col("close") / F.lag("close", 24).over(w)) - F.lit(1.0)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Moving Averages & Ratios

# COMMAND ----------

for window_size in [7, 24, 168]:
    features = features.withColumn(
        f"ma_{window_size}",
        F.avg("close").over(w.rowsBetween(-window_size, -1)),
    )

# MA Ratios — scale-invariant, ổn định hơn giá tuyệt đối
features = features.withColumn(
    "close_ma7_ratio", F.col("close") / F.col("ma_7")
)
features = features.withColumn(
    "close_ma24_ratio", F.col("close") / F.col("ma_24")
)
features = features.withColumn(
    "close_ma168_ratio", F.col("close") / F.col("ma_168")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. MACD (Moving Average Convergence Divergence)
# MAGIC
# MAGIC Sử dụng EMA xấp xỉ qua Exponential Weighted Moving Average trên PySpark.
# MAGIC Vì PySpark không có hàm EMA built-in, ta tính xấp xỉ bằng cách convert
# MAGIC EMA span sang alpha rồi dùng window functions.

# COMMAND ----------

# Tính EMA xấp xỉ bằng pandas UDF cho chính xác hơn
from pyspark.sql.types import DoubleType
import pyspark.sql.functions as F

# SMA12, SMA26 dùng rolling window.

# Approximate EMA using simple moving average (SMA) — đủ tốt cho tree-based models
sma_12_window = w.rowsBetween(-12, -1)
sma_26_window = w.rowsBetween(-26, -1)
ema_9_window = w.rowsBetween(-9, -1)

features = features.withColumn("sma_12", F.avg("close").over(sma_12_window))
features = features.withColumn("sma_26", F.avg("close").over(sma_26_window))
features = features.withColumn("macd", F.col("sma_12") - F.col("sma_26"))
features = features.withColumn("macd_signal", F.avg("macd").over(ema_9_window))
features = features.withColumn("macd_hist", F.col("macd") - F.col("macd_signal"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. RSI (Relative Strength Index) — 14 periods

# COMMAND ----------

# RSI = 100 - (100 / (1 + RS))
# RS = avg_gain / avg_loss over 14 periods

features = features.withColumn(
    "price_change", F.col("close") - F.col("close_lag_1h")
)
features = features.withColumn(
    "gain", F.when(F.col("price_change") > 0, F.col("price_change")).otherwise(0.0)
)
features = features.withColumn(
    "loss", F.when(F.col("price_change") < 0, -F.col("price_change")).otherwise(0.0)
)

rsi_window = w.rowsBetween(-14, -1)
features = features.withColumn("avg_gain", F.avg("gain").over(rsi_window))
features = features.withColumn("avg_loss", F.avg("loss").over(rsi_window))

features = features.withColumn(
    "rsi_14",
    F.when(
        F.col("avg_loss") == 0, F.lit(100.0)
    ).otherwise(
        F.lit(100.0) - (F.lit(100.0) / (F.lit(1.0) + F.col("avg_gain") / F.col("avg_loss")))
    ),
)

# Dọn dẹp cột tạm
features = features.drop("price_change", "gain", "loss", "avg_gain", "avg_loss")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. ATR (Average True Range) & Volatility

# COMMAND ----------

# True Range = max(high - low, |high - prev_close|, |low - prev_close|)
features = features.withColumn("prev_close", F.col("close_lag_1h"))
features = features.withColumn(
    "true_range",
    F.greatest(
        F.col("high") - F.col("low"),
        F.abs(F.col("high") - F.col("prev_close")),
        F.abs(F.col("low") - F.col("prev_close")),
    ),
)

atr_window = w.rowsBetween(-14, -1)
features = features.withColumn("atr_14", F.avg("true_range").over(atr_window))
features = features.withColumn("atr_ratio", F.col("atr_14") / F.col("close"))

# Bollinger Band Width = (upper - lower) / middle
# upper = MA20 + 2*std, lower = MA20 - 2*std
bb_window = w.rowsBetween(-20, -1)
features = features.withColumn("bb_ma20", F.avg("close").over(bb_window))
features = features.withColumn("bb_std20", F.stddev("close").over(bb_window))
features = features.withColumn(
    "bb_width",
    (F.lit(4.0) * F.col("bb_std20")) / F.col("bb_ma20"),  # (upper-lower)/middle = 4*std/ma
)

# Dọn dẹp cột tạm
features = features.drop("prev_close", "true_range", "bb_ma20", "bb_std20")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Volume Features

# COMMAND ----------

volume_ma24_window = w.rowsBetween(-24, -1)
features = features.withColumn(
    "volume_ma24", F.avg("volume").over(volume_ma24_window)
)
features = features.withColumn(
    "volume_ratio", F.col("volume") / F.col("volume_ma24")
)
features = features.withColumn(
    "log_volume", F.log1p("volume")
)

# Dọn dẹp cột tạm
features = features.drop("volume_ma24")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Cyclical Time Features

# COMMAND ----------

import math

features = features.withColumn("hour", F.hour("open_time"))
features = features.withColumn("day_of_week", F.dayofweek("open_time"))

# Cyclical encoding — giúp model hiểu 23h gần 0h, Chủ nhật gần Thứ 2
features = features.withColumn(
    "hour_sin", F.sin(F.lit(2.0 * math.pi) * F.col("hour") / F.lit(24.0))
)
features = features.withColumn(
    "hour_cos", F.cos(F.lit(2.0 * math.pi) * F.col("hour") / F.lit(24.0))
)
features = features.withColumn(
    "weekday_sin", F.sin(F.lit(2.0 * math.pi) * F.col("day_of_week") / F.lit(7.0))
)
features = features.withColumn(
    "weekday_cos", F.cos(F.lit(2.0 * math.pi) * F.col("day_of_week") / F.lit(7.0))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Volatility Regime & Momentum Features

# COMMAND ----------

# --- Volatility Regime: rolling std of returns ---
for vol_window in [12, 24, 168]:
    vol_w = w.rowsBetween(-vol_window, -1)
    features = features.withColumn(
        f"volatility_{vol_window}h",
        F.stddev("return_1h").over(vol_w),
    )

# Volatility z-score: current vol relative to 168h baseline
features = features.withColumn(
    "volatility_zscore",
    (F.col("volatility_24h") - F.avg("volatility_24h").over(w.rowsBetween(-168, -1)))
    / F.stddev("volatility_24h").over(w.rowsBetween(-168, -1)),
)

# --- Momentum: Rate of Change (ROC) ---
for roc_period in [3, 6, 12]:
    features = features.withColumn(
        f"roc_{roc_period}h",
        (F.col("close") - F.lag("close", roc_period).over(w))
        / F.lag("close", roc_period).over(w),
    )

# --- Price Acceleration: second derivative of price ---
features = features.withColumn(
    "price_acceleration",
    F.col("close") - 2 * F.lag("close", 1).over(w) + F.lag("close", 2).over(w),
)

# --- Volume-Price Divergence ---
# Compare volume trend (24h) vs price trend (24h)
features = features.withColumn(
    "volume_trend_24h",
    (F.col("volume") - F.lag("volume", 24).over(w)) / F.lag("volume", 24).over(w),
)
features = features.withColumn(
    "price_trend_24h",
    (F.col("close") - F.lag("close", 24).over(w)) / F.lag("close", 24).over(w),
)
features = features.withColumn(
    "vol_price_divergence",
    F.col("volume_trend_24h") - F.col("price_trend_24h"),
)

# --- Return Skewness & Kurtosis approximations (168h window) ---
# Photon doesn't optimize skewness/kurtosis over windows well.
# Use simpler asymmetry and tail-weight proxies instead.
dist_window = w.rowsBetween(-168, -1)
features = features.withColumn(
    "return_asymmetry_168h",
    (F.col("return_1h") - F.avg("return_1h").over(dist_window))
    / F.stddev("return_1h").over(dist_window),
)
features = features.withColumn(
    "return_tail_weight_168h",
    F.abs(F.col("return_1h") - F.avg("return_1h").over(dist_window))
    / F.stddev("return_1h").over(dist_window),
)

# --- Intraday Volatility Pattern: ratio of recent vol to long-term vol ---
features = features.withColumn(
    "vol_ratio_12_168",
    F.col("volatility_12h") / F.col("volatility_168h"),
)

# Dọn cột tạm
features = features.drop("volume_trend_24h", "price_trend_24h")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Targets

# COMMAND ----------

# Exact next-hour target: do not use next-row lead because missing candles would mislabel targets.
targets = raw.select(
    F.col("open_time").alias("target_open_time"),
    F.col("close").alias("target_close_1h"),
)
features = features.join(
    targets,
    F.col("target_open_time") == F.col("open_time") + F.expr("INTERVAL 1 HOUR"),
    "left",
).drop("target_open_time")
features = features.withColumn(
    "target_return_1h",
    (F.col("target_close_1h") / F.col("close")) - F.lit(1.0),
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Final Cleanup & Save

# COMMAND ----------

# Dọn các cột EMA tạm (giữ lại macd, macd_signal, macd_hist)
features = features.drop("sma_12", "sma_26")

lookback_required_cols = [
    "close_lag_1h",
    "close_lag_2h",
    "close_lag_4h",
    "close_lag_12h",
    "close_lag_24h",
    "return_1h",
    "return_6h",
    "return_24h",
    "close_ma7_ratio",
    "close_ma24_ratio",
    "close_ma168_ratio",
    "macd",
    "macd_signal",
    "macd_hist",
    "rsi_14",
    "atr_14",
    "atr_ratio",
    "bb_width",
    "volume_ratio",
    "volatility_24h",
    "volatility_zscore",
    "roc_3h",
    "roc_6h",
    "roc_12h",
    "vol_price_divergence",
    "return_asymmetry_168h",
    "return_tail_weight_168h",
    "vol_ratio_12_168",
]
pre_lookback_drop_count = features.count()
features = features.dropna(subset=lookback_required_cols)
post_lookback_drop_count = features.count()
print(f"dropped_null_lookback_rows={pre_lookback_drop_count - post_lookback_drop_count}")

feature_count = features.count()
print(f"feature_count={feature_count}")

# In danh sách tất cả columns
print(f"columns={features.columns}")

# COMMAND ----------

features.createOrReplaceTempView("_btc_features_upsert")

try:
    spark.table(features_ref).limit(1).collect()
    features_table_exists = True
except Exception as exc:
    print(f"features_table_missing={exc}")
    features_table_exists = False

if features_table_exists:
    features.limit(0).write.format("delta").mode("append").option(
        "mergeSchema", "true"
    ).saveAsTable(features_ref)
    cols = ", ".join([f"target.`{c}` = source.`{c}`" for c in features.columns])
    insert_cols = ", ".join([f"`{c}`" for c in features.columns])
    insert_vals = ", ".join([f"source.`{c}`" for c in features.columns])
    spark.sql(f"""
        MERGE INTO {features_ref} AS target
        USING _btc_features_upsert AS source
        ON target.`symbol` <=> source.`symbol`
        AND target.`open_time` = source.`open_time`
        WHEN MATCHED THEN UPDATE SET {cols}
        WHEN NOT MATCHED THEN INSERT ({insert_cols}) VALUES ({insert_vals})
    """)
else:
    features.write.format("delta").mode("overwrite").saveAsTable(features_ref)

# COMMAND ----------

result = spark.table(features_ref)
print(f"features_table_count={result.count()}")
print(f"null_target_return_1h={result.filter(F.col('target_return_1h').isNull()).count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Feature Selection Config

# COMMAND ----------

candidate_features = [
    "return_1h", "return_6h", "return_24h",
    "ma_7", "ma_24", "ma_168",
    "close_ma7_ratio", "close_ma24_ratio", "close_ma168_ratio",
    "macd", "macd_signal", "macd_hist",
    "rsi_14",
    "atr_14", "atr_ratio", "bb_width",
    "volume_ratio", "log_volume",
    "hl_spread", "oc_change",
    "close_lag_1h", "close_lag_2h", "close_lag_4h", "close_lag_12h", "close_lag_24h",
    "hour", "day_of_week",
    "hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
    "volatility_12h", "volatility_24h", "volatility_168h",
    "volatility_zscore",
    "roc_3h", "roc_6h", "roc_12h",
    "price_acceleration",
    "vol_price_divergence",
    "return_asymmetry_168h",
    "return_tail_weight_168h",
    "vol_ratio_12_168",
]
target_col = "target_return_1h"
features_table_version = int(spark.sql(f"DESCRIBE HISTORY {features_ref} LIMIT 1").collect()[0]["version"])
available_features = [column for column in candidate_features if column in result.columns]
missing_features = [column for column in candidate_features if column not in result.columns]
if missing_features:
    print(f"feature_selection_missing_columns={missing_features}")

selection_pdf = (
    result.select(["open_time", target_col] + available_features)
    .orderBy("open_time")
    .toPandas()
    .dropna(subset=[target_col])
)
selection_pdf = selection_pdf[available_features + [target_col]].dropna()
print(f"feature_selection_rows={len(selection_pdf)}")
if len(selection_pdf) < 6:
    print("feature_selection_skip=too_few_samples")
    fallback_feature_candidates = [
        "return_1h", "return_6h", "return_24h",
        "close_ma7_ratio", "close_ma24_ratio", "close_ma168_ratio",
        "macd", "macd_signal", "macd_hist",
        "rsi_14", "atr_14", "atr_ratio", "bb_width",
        "volume_ratio", "log_volume", "hl_spread", "oc_change",
        "close_lag_1h", "close_lag_2h", "close_lag_4h", "close_lag_12h", "close_lag_24h",
        "hour", "day_of_week", "hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
        "volatility_12h", "volatility_24h", "volatility_168h",
        "volatility_zscore",
        "roc_3h", "roc_6h", "roc_12h",
        "price_acceleration",
        "vol_price_divergence",
        "return_asymmetry_168h", "return_tail_weight_168h",
        "vol_ratio_12_168",
    ]
    selected_features = [column for column in fallback_feature_candidates if column in available_features]
    if not selected_features:
        raise ValueError("Fallback feature selection produced an empty selected_features list")
    features_to_drop = set(available_features) - set(selected_features)
    created_at = pd.Timestamp.now(tz="UTC")
    config_id = int(created_at.timestamp() * 1000)
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {feature_config_ref} (
            config_key STRING,
            config_value STRING,
            config_id BIGINT,
            config_version BIGINT,
            created_at STRING,
            created_by STRING,
            is_active BOOLEAN,
            n_features BIGINT,
            method STRING,
            source_table STRING,
            source_table_version BIGINT,
            target_col STRING,
            candidate_features_json STRING,
            dropped_features_json STRING,
            selection_metrics_json STRING,
            corr_threshold DOUBLE,
            mi_threshold DOUBLE
        )
        USING DELTA
    """)
    config_df = spark.createDataFrame([{
        "config_key": "selected_features",
        "config_value": json.dumps(selected_features),
        "config_id": config_id,
        "config_version": config_id,
        "created_at": created_at.isoformat(),
        "created_by": "02_feature_engineering",
        "is_active": True,
        "n_features": len(selected_features),
        "method": "fallback_too_few_samples",
        "source_table": features_ref,
        "source_table_version": features_table_version,
        "target_col": target_col,
        "candidate_features_json": json.dumps(available_features),
        "dropped_features_json": json.dumps(sorted(features_to_drop)),
        "selection_metrics_json": json.dumps({"feature_selection_rows": len(selection_pdf)}),
        "corr_threshold": corr_threshold,
        "mi_threshold": mi_threshold,
    }])
    config_df.write.format("delta").mode("append").option(
        "mergeSchema", "true"
    ).saveAsTable(feature_config_ref)
    spark.sql(f"""
        UPDATE {feature_config_ref}
        SET is_active = false
        WHERE config_key = 'selected_features'
          AND is_active = true
          AND COALESCE(config_id, config_version) != {config_id}
    """)
    print(f"fallback_feature_config_saved={feature_config_ref}")
    print(f"feature_config_id={config_id}")
    dbutils.notebook.exit("SKIPPED: Not enough samples for feature selection")

X = selection_pdf[available_features]
y = selection_pdf[target_col]

corr_matrix = X.corr(method="pearson")
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        corr_value = corr_matrix.iloc[i, j]
        if pd.isna(corr_value):
            continue
        abs_corr = abs(corr_value)
        if abs_corr >= corr_threshold:
            high_corr_pairs.append({
                "feature_1": corr_matrix.columns[i],
                "feature_2": corr_matrix.columns[j],
                "correlation": round(float(corr_value), 4),
                "abs_correlation": round(float(abs_corr), 4),
            })
high_corr_df = pd.DataFrame(
    high_corr_pairs,
    columns=["feature_1", "feature_2", "correlation", "abs_correlation"],
).sort_values("abs_correlation", ascending=False)

n_neighbors = min(5, max(1, len(X) - 1))
mi_values = mutual_info_regression(X, y, random_state=42, n_neighbors=n_neighbors)
mi_df = pd.DataFrame({
    "feature": available_features,
    "mi_regression": mi_values,
}).sort_values("mi_regression", ascending=False)

lgbm = LGBMRegressor(n_estimators=100, max_depth=6, random_state=42, verbose=-1)
lgbm.fit(X, y)
importance_df = pd.DataFrame({
    "feature": available_features,
    "importance_regression": lgbm.feature_importances_,
}).sort_values("importance_regression", ascending=False)

ranking = pd.DataFrame({"feature": available_features})
ranking = ranking.merge(mi_df, on="feature", how="left")
ranking = ranking.merge(importance_df, on="feature", how="left")
for metric_col in ["mi_regression", "importance_regression"]:
    ranking[f"rank_{metric_col}"] = ranking[metric_col].rank(ascending=False)
rank_cols = [column for column in ranking.columns if column.startswith("rank_")]
ranking["avg_rank"] = ranking[rank_cols].mean(axis=1)
ranking = ranking.sort_values("avg_rank")

features_to_drop_corr = set()
for _, row in high_corr_df.iterrows():
    feature_1 = row["feature_1"]
    feature_2 = row["feature_2"]
    if feature_1 in features_to_drop_corr or feature_2 in features_to_drop_corr:
        continue
    rank_1 = ranking.loc[ranking["feature"] == feature_1, "avg_rank"].values
    rank_2 = ranking.loc[ranking["feature"] == feature_2, "avg_rank"].values
    if len(rank_1) > 0 and len(rank_2) > 0:
        if rank_1[0] <= rank_2[0]:
            features_to_drop_corr.add(feature_2)
        else:
            features_to_drop_corr.add(feature_1)

low_mi_features = set(ranking[ranking["mi_regression"] < mi_threshold]["feature"].tolist())
features_to_drop = features_to_drop_corr | low_mi_features
selected_features = [feature for feature in available_features if feature not in features_to_drop]
if not selected_features:
    raise ValueError("Feature selection produced an empty selected_features list")

print(f"selected_features_count={len(selected_features)}")
print(f"selected_features={selected_features}")
print(f"dropped_features={sorted(features_to_drop)}")

created_at = pd.Timestamp.now(tz="UTC")
config_id = int(created_at.timestamp() * 1000)

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {feature_config_ref} (
        config_key STRING,
        config_value STRING,
        config_id BIGINT,
        config_version BIGINT,
        created_at STRING,
        created_by STRING,
        is_active BOOLEAN,
        n_features BIGINT,
        method STRING,
        source_table STRING,
        source_table_version BIGINT,
        target_col STRING,
        candidate_features_json STRING,
        dropped_features_json STRING,
        selection_metrics_json STRING,
        corr_threshold DOUBLE,
        mi_threshold DOUBLE
    )
    USING DELTA
""")

for column_spec in [
    "config_id BIGINT",
    "created_by STRING",
    "is_active BOOLEAN",
    "source_table STRING",
    "source_table_version BIGINT",
    "target_col STRING",
    "candidate_features_json STRING",
    "dropped_features_json STRING",
    "selection_metrics_json STRING",
    "corr_threshold DOUBLE",
    "mi_threshold DOUBLE",
]:
    try:
        spark.sql(f"ALTER TABLE {feature_config_ref} ADD COLUMNS ({column_spec})")
    except Exception as exc:
        print(f"feature_config_column_add_skipped={column_spec}: {exc}")

selection_metrics = {
    "top_mi_features": mi_df.head(20).to_dict(orient="records"),
    "top_importance_features": importance_df.head(20).to_dict(orient="records"),
    "high_corr_pairs": high_corr_df.head(50).to_dict(orient="records") if len(high_corr_df) else [],
}

config_df = spark.createDataFrame([{
    "config_key": "selected_features",
    "config_value": json.dumps(selected_features),
    "config_id": config_id,
    "config_version": config_id,
    "created_at": created_at.isoformat(),
    "created_by": "02_feature_engineering",
    "is_active": True,
    "n_features": len(selected_features),
    "method": "feature_engineering_auto_selection",
    "source_table": features_ref,
    "source_table_version": features_table_version,
    "target_col": target_col,
    "candidate_features_json": json.dumps(available_features),
    "dropped_features_json": json.dumps(sorted(features_to_drop)),
    "selection_metrics_json": json.dumps(selection_metrics, default=str),
    "corr_threshold": corr_threshold,
    "mi_threshold": mi_threshold,
}])

config_df.write.format("delta").mode("append").option(
    "mergeSchema", "true"
).saveAsTable(feature_config_ref)

spark.sql(f"""
    UPDATE {feature_config_ref}
    SET is_active = false
    WHERE config_key = 'selected_features'
      AND is_active = true
      AND COALESCE(config_id, config_version) != {config_id}
""")

print(f"feature_config_saved={feature_config_ref}")
print(f"feature_config_id={config_id}")

# COMMAND ----------

# Thống kê cơ bản cho các feature mới
display(result.select(
    "return_1h", "return_6h", "return_24h",
    "close_ma7_ratio", "close_ma24_ratio", "close_ma168_ratio",
    "macd", "macd_signal", "macd_hist",
    "rsi_14", "atr_14", "atr_ratio", "bb_width",
    "volume_ratio", "log_volume",
    "target_return_1h",
).summary())

# COMMAND ----------

display(result.orderBy("open_time").limit(10))

# COMMAND ----------

display(result.orderBy(F.col("open_time").desc()).limit(10))
