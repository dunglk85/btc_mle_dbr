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

from pyspark.sql import Window, functions as F

# COMMAND ----------

def get_widget(name, default):
    try:
        dbutils.widgets.text(name, str(default))
        return dbutils.widgets.get(name)
    except Exception:
        return str(default)


catalog = get_widget("catalog", "btc_dev")
raw_schema = "raw"
features_schema = "features"
raw_table = "btc_hourly"
features_table = "btc_features"
raw_ref = f"{catalog}.{raw_schema}.{raw_table}"
features_ref = f"{catalog}.{features_schema}.{features_table}"

print("RUNNING ADVANCED FEATURE ENGINEERING NOTEBOOK")
print(f"raw_ref={raw_ref}")
print(f"features_ref={features_ref}")

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{features_schema}")

# COMMAND ----------

raw = spark.table(raw_ref).dropDuplicates(["open_time"]).orderBy("open_time")
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
# MAGIC ## 9. Targets

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
    spark.sql(f"""
        MERGE INTO {features_ref} AS target
        USING _btc_features_upsert AS source
        ON target.symbol <=> source.symbol
        AND target.open_time = source.open_time
        WHEN MATCHED THEN UPDATE SET *
        WHEN NOT MATCHED THEN INSERT *
    """)
else:
    features.write.format("delta").mode("errorifexists").saveAsTable(features_ref)

# COMMAND ----------

result = spark.table(features_ref)
print(f"features_table_count={result.count()}")
print(f"null_target_return_1h={result.filter(F.col('target_return_1h').isNull()).count()}")

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
