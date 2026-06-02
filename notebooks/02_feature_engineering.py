# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 02 - Feature Engineering

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

print("RUNNING SELF-CONTAINED FEATURE ENGINEERING NOTEBOOK")
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

w = Window.orderBy("open_time")
features = raw

for window_size in [7, 24, 168]:
    features = features.withColumn(
        f"ma_{window_size}",
        F.avg("close").over(w.rowsBetween(-window_size, -1)),
    )

for lag_hour in [1, 2, 4, 12, 24]:
    features = features.withColumn(
        f"close_lag_{lag_hour}h",
        F.lag("close", lag_hour).over(w),
    )

features = features.withColumn(
    "return_1h", (F.col("close") / F.lag("close", 1).over(w)) - F.lit(1.0)
)
features = features.withColumn("hl_spread", F.col("high") - F.col("low"))
features = features.withColumn("oc_change", F.col("close") - F.col("open"))
features = features.withColumn("hour", F.hour("open_time"))
features = features.withColumn("day_of_week", F.dayofweek("open_time"))
target = raw.select(
    (F.col("open_time") - F.expr("INTERVAL 1 HOUR")).alias("open_time"),
    F.col("close").alias("target_close_1h"),
)
features = features.join(target, on="open_time", how="left")

feature_count = features.count()
print(f"feature_count={feature_count}")

# COMMAND ----------

features.write.format("delta").mode("overwrite").option(
    "overwriteSchema", "true"
).saveAsTable(features_ref)

# COMMAND ----------

result = spark.table(features_ref)
print(f"features_table_count={result.count()}")

# COMMAND ----------

display(result.orderBy("open_time").limit(10))

# COMMAND ----------

display(result.orderBy(F.col("open_time").desc()).limit(10))
