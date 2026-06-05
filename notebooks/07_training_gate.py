# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 07 - Training Gate

# COMMAND ----------

from datetime import datetime, timezone

import mlflow
from mlflow.tracking import MlflowClient
from pyspark.sql import Window, functions as F

# COMMAND ----------

def get_widget(name, default):
    try:
        dbutils.widgets.text(name, str(default))
        return dbutils.widgets.get(name)
    except Exception:
        return str(default)


catalog = get_widget("catalog", "btc_dev")
monitoring_schema = "monitoring"
model_schema = "models"
model_name = "btc_price_model"

metrics_ref = f"{catalog}.{monitoring_schema}.pipeline_metrics"
decisions_ref = f"{catalog}.{monitoring_schema}.model_refresh_decisions"
full_model_name = f"{catalog}.{model_schema}.{model_name}"

trigger_mode = get_widget("trigger_mode", "scheduled")
max_raw_freshness_hours = float(get_widget("max_raw_freshness_hours", "3"))

print("RUNNING SELF-CONTAINED TRAINING GATE NOTEBOOK")
print(f"metrics_ref={metrics_ref}")
print(f"decisions_ref={decisions_ref}")
print(f"full_model_name={full_model_name}")
print(f"trigger_mode={trigger_mode}")

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{monitoring_schema}")
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {decisions_ref} (
        decision_time TIMESTAMP,
        should_retrain BOOLEAN,
        reason STRING,
        trigger_mode STRING,
        raw_freshness_hours DOUBLE,
        alert_count BIGINT,
        champion_exists BOOLEAN
    )
    USING DELTA
""")

# COMMAND ----------

metrics_exists = True
try:
    metrics = spark.table(metrics_ref)
    metrics.limit(1).collect()
except Exception as exc:
    print(f"metrics table unavailable: {exc}")
    metrics_exists = False

raw_freshness_hours = None
alert_count = 0
blocking_alert_count = 0
drift_alert_count = 0
validation_metric_count = 0
if metrics_exists:
    latest_window = Window.partitionBy("metric_name").orderBy(F.col("metric_time").desc())
    latest_metrics = metrics.withColumn("_rn", F.row_number().over(latest_window)).filter(
        F.col("_rn") == 1
    )
    drift_metric = F.col("metric_name").rlike(
        "^(data_drift|model_drift|prediction_drift|label_drift|concept_drift)_"
    )
    retrain_drift_metric = F.col("metric_name").rlike("^(model_drift|concept_drift)_")
    validation_metric = F.col("metric_name").rlike(
        "^(raw_count|raw_duplicate_open_time_count|raw_null_open_time_count|"
        "raw_freshness_hours|features_count|features_target_close_1h_null_count|"
        "raw_features_row_count_delta|feature_quality_|schema_drift_)"
    )
    alert_count = latest_metrics.filter(F.col("status") == "alert").count()
    drift_alert_count = latest_metrics.filter(
        (F.col("status") == "alert") & retrain_drift_metric
    ).count()
    blocking_alert_count = latest_metrics.filter(
        (F.col("status") == "alert") & (~drift_metric)
    ).count()
    validation_metric_count = latest_metrics.filter(validation_metric).count()
    raw_freshness = latest_metrics.filter(
        F.col("metric_name") == "raw_freshness_hours"
    ).select("metric_value").collect()
    if raw_freshness:
        raw_freshness_hours = raw_freshness[0]["metric_value"]

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()
champion_exists = True
try:
    champion = client.get_model_version_by_alias(full_model_name, "Champion")
    print(f"champion_version={champion.version}")
except Exception as exc:
    print(f"No Champion alias found: {exc}")
    champion_exists = False

# COMMAND ----------

reasons = []
block_reasons = []
should_retrain = trigger_mode == "scheduled"

if blocking_alert_count > 0:
    should_retrain = False
    block_reasons.append(f"latest monitoring has {blocking_alert_count} blocking alert metrics")

if trigger_mode == "drift" and validation_metric_count == 0:
    should_retrain = False
    block_reasons.append("missing data/schema/feature quality validation metrics")

if raw_freshness_hours is not None and raw_freshness_hours > max_raw_freshness_hours:
    should_retrain = False
    block_reasons.append(
        f"raw data stale: {raw_freshness_hours:.2f}h > {max_raw_freshness_hours:.2f}h"
    )

if block_reasons:
    reasons.extend(block_reasons)

if drift_alert_count > 0 and blocking_alert_count == 0 and validation_metric_count > 0:
    should_retrain = True
    reasons.append(
        f"drift detected: {drift_alert_count} alert metrics; "
        "data/schema/feature quality validation passed"
    )

if not champion_exists and blocking_alert_count == 0:
    should_retrain = True
    reasons.append("no Champion exists")

if trigger_mode == "manual" and blocking_alert_count == 0:
    should_retrain = True
    reasons.append(f"trigger_mode={trigger_mode}")

if trigger_mode == "drift" and not reasons:
    reasons.append("no drift trigger detected")

if not reasons:
    reasons.append("scheduled refresh allowed")

if should_retrain:
    decision_status = "RETRAIN_ALLOWED"
elif block_reasons:
    decision_status = "NO_RETRAIN_BLOCKED"
else:
    decision_status = "NO_RETRAIN_NO_TRIGGER"

reason = "; ".join(reasons)
print(f"decision_status={decision_status}")
print(f"should_retrain={should_retrain}")
print(f"reason={reason}")

# COMMAND ----------

decision_df = spark.createDataFrame(
    [
        {
            "decision_time": datetime.now(timezone.utc),
            "should_retrain": bool(should_retrain),
            "reason": reason,
            "trigger_mode": trigger_mode,
            "raw_freshness_hours": raw_freshness_hours,
            "alert_count": int(alert_count),
            "champion_exists": bool(champion_exists),
        }
    ]
)
decision_df.write.mode("append").saveAsTable(decisions_ref)

display(decision_df)
