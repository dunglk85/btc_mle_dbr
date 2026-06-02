# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 07 - Monitoring Gate

# COMMAND ----------

from datetime import datetime, timezone

import mlflow
from mlflow.tracking import MlflowClient
from pyspark.sql import functions as F

# COMMAND ----------

catalog = "btc_dev"
monitoring_schema = "monitoring"
model_schema = "models"
model_name = "btc_price_model"

metrics_ref = f"{catalog}.{monitoring_schema}.pipeline_metrics"
decisions_ref = f"{catalog}.{monitoring_schema}.model_refresh_decisions"
full_model_name = f"{catalog}.{model_schema}.{model_name}"


def get_widget(name, default):
    try:
        dbutils.widgets.text(name, str(default))
        return dbutils.widgets.get(name)
    except Exception:
        return str(default)


trigger_mode = get_widget("trigger_mode", "scheduled")
max_raw_freshness_hours = float(get_widget("max_raw_freshness_hours", "3"))

print("RUNNING SELF-CONTAINED MONITORING GATE NOTEBOOK")
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
if metrics_exists:
    latest_metric_time = metrics.agg(F.max("metric_time").alias("max_time")).collect()[0][
        "max_time"
    ]
    if latest_metric_time is not None:
        latest_metrics = metrics.filter(F.col("metric_time") == latest_metric_time)
        alert_count = latest_metrics.filter(F.col("status") == "alert").count()
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
should_retrain = True

if alert_count > 0:
    should_retrain = False
    reasons.append(f"latest monitoring has {alert_count} alert metrics")

if raw_freshness_hours is not None and raw_freshness_hours > max_raw_freshness_hours:
    should_retrain = False
    reasons.append(
        f"raw data stale: {raw_freshness_hours:.2f}h > {max_raw_freshness_hours:.2f}h"
    )

if not champion_exists and alert_count == 0:
    should_retrain = True
    reasons.append("no Champion exists")

if trigger_mode in ("manual", "drift") and alert_count == 0:
    should_retrain = True
    reasons.append(f"trigger_mode={trigger_mode}")

if not reasons:
    reasons.append("scheduled refresh allowed")

reason = "; ".join(reasons)
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
