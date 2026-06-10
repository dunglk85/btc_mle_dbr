# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 06 - Monitoring

# COMMAND ----------

import json
from datetime import datetime, timezone

from pyspark.sql import functions as F

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
predictions_schema = "predictions"
monitoring_schema = "monitoring"

raw_ref = f"{catalog}.{raw_schema}.btc_hourly"
features_ref = f"{catalog}.{features_schema}.btc_features"
predictions_ref = f"{catalog}.{predictions_schema}.btc_predictions"
metrics_ref = f"{catalog}.{monitoring_schema}.pipeline_metrics"

freshness_threshold_hours = 3

print("RUNNING SELF-CONTAINED MONITORING NOTEBOOK")
print(f"raw_ref={raw_ref}")
print(f"features_ref={features_ref}")
print(f"predictions_ref={predictions_ref}")
print(f"metrics_ref={metrics_ref}")

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{monitoring_schema}")
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {metrics_ref} (
        metric_time TIMESTAMP,
        metric_name STRING,
        metric_value DOUBLE,
        status STRING,
        details STRING
    )
    USING DELTA
""")

# COMMAND ----------


def table_exists(table_ref):
    try:
        spark.table(table_ref).limit(1).collect()
        return True
    except Exception:
        return False


def latest_table_version(table_ref):
    try:
        row = spark.sql(f"DESCRIBE HISTORY {table_ref} LIMIT 1").collect()[0]
        return {"table": table_ref, "version": int(row["version"]), "timestamp": str(row["timestamp"])}
    except Exception as exc:
        return {"table": table_ref, "version": None, "error": str(exc)}


metric_time = datetime.now(timezone.utc)
lineage_context = {
    "raw": latest_table_version(raw_ref),
    "features": latest_table_version(features_ref),
    "predictions": latest_table_version(predictions_ref) if table_exists(predictions_ref) else None,
}


def append_metric(metrics, name, value, status="ok", details=""):
    details_payload = {"message": details, "lineage": lineage_context}
    metrics.append(
        {
            "metric_time": metric_time,
            "metric_name": name,
            "metric_value": float(value) if value is not None else None,
            "status": status,
            "details": json.dumps(details_payload),
        }
    )


metrics = []

# COMMAND ----------

raw = spark.table(raw_ref)
raw_count = raw.count()
latest_raw = raw.agg(F.max("open_time").alias("latest_raw")).collect()[0]["latest_raw"]
duplicate_raw_count = raw.groupBy("open_time").count().filter(F.col("count") > 1).count()
null_raw_open_time_count = raw.filter(F.col("open_time").isNull()).count()

append_metric(metrics, "raw_count", raw_count)
append_metric(
    metrics,
    "raw_duplicate_open_time_count",
    duplicate_raw_count,
    "ok" if duplicate_raw_count == 0 else "alert",
)
append_metric(
    metrics,
    "raw_null_open_time_count",
    null_raw_open_time_count,
    "ok" if null_raw_open_time_count == 0 else "alert",
)

if latest_raw is None:
    append_metric(metrics, "raw_freshness_hours", None, "alert", "No raw data")
else:
    latest_raw_utc = latest_raw.replace(tzinfo=timezone.utc)
    freshness_hours = (datetime.now(timezone.utc) - latest_raw_utc).total_seconds() / 3600
    append_metric(
        metrics,
        "raw_freshness_hours",
        freshness_hours,
        "ok" if freshness_hours <= freshness_threshold_hours else "alert",
        f"latest_raw={latest_raw}",
    )

# COMMAND ----------

features = spark.table(features_ref)
features_count = features.count()
features_target_null_count = features.filter(F.col("target_close_1h").isNull()).count()

append_metric(metrics, "features_count", features_count)
append_metric(
    metrics,
    "features_target_close_1h_null_count",
    features_target_null_count,
    "ok" if features_target_null_count <= 1 else "alert",
    "Expected only rows without exact next-hour candles to have null target.",
)
append_metric(
    metrics,
    "raw_features_row_count_delta",
    raw_count - features_count,
    "ok" if abs(raw_count - features_count) <= 1 else "alert",
)

# COMMAND ----------

if table_exists(predictions_ref):
    predictions = spark.table(predictions_ref)
    prediction_count = predictions.count()
    latest_prediction_time = predictions.agg(
        F.max("prediction_time").alias("latest_prediction_time")
    ).collect()[0]["latest_prediction_time"]
    append_metric(metrics, "prediction_count", prediction_count)
    if latest_prediction_time is not None:
        latest_prediction_utc = latest_prediction_time.replace(tzinfo=timezone.utc)
        prediction_age_hours = (
            datetime.now(timezone.utc) - latest_prediction_utc
        ).total_seconds() / 3600
        append_metric(
            metrics,
            "prediction_age_hours",
            prediction_age_hours,
            "ok" if prediction_age_hours <= freshness_threshold_hours else "alert",
            f"latest_prediction_time={latest_prediction_time}",
        )
    else:
        append_metric(metrics, "prediction_age_hours", None, "alert", "No predictions")
else:
    append_metric(metrics, "prediction_count", 0, "warn", "Predictions table does not exist")

# COMMAND ----------

metrics_df = spark.createDataFrame(metrics)
metrics_df.write.mode("append").saveAsTable(metrics_ref)

alert_count = metrics_df.filter(F.col("status") == "alert").count()
warn_count = metrics_df.filter(F.col("status") == "warn").count()

print(f"metrics_written={metrics_df.count()}")
print(f"alert_count={alert_count}")
print(f"warn_count={warn_count}")

display(metrics_df.orderBy("metric_name"))

if alert_count > 0:
    raise ValueError(f"Monitoring produced {alert_count} alert metrics")
