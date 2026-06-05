# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 10 - Data Remediation
# MAGIC
# MAGIC Auto-remediate safe data issues. Dangerous issues are logged for manual handling.

# COMMAND ----------

from datetime import datetime, timezone
import json

from pyspark.sql import Window, functions as F

# COMMAND ----------

def get_widget(name, default):
    try:
        dbutils.widgets.text(name, str(default))
        return dbutils.widgets.get(name)
    except Exception:
        return str(default)


catalog = get_widget("catalog", "btc_dev")
backfill_limit = get_widget("backfill_limit", "168")
notebook_timeout_seconds = int(get_widget("notebook_timeout_seconds", "1800"))

monitoring_schema = "monitoring"
metrics_ref = f"{catalog}.{monitoring_schema}.pipeline_metrics"
actions_ref = f"{catalog}.{monitoring_schema}.data_remediation_actions"

print("RUNNING DATA REMEDIATION NOTEBOOK")
print(f"metrics_ref={metrics_ref}")
print(f"actions_ref={actions_ref}")
print(f"backfill_limit={backfill_limit}")

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{monitoring_schema}")
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {actions_ref} (
        action_time TIMESTAMP,
        status STRING,
        reason STRING,
        action_type STRING,
        metrics_snapshot STRING
    )
    USING DELTA
""")

# COMMAND ----------

def latest_metric_rows():
    try:
        metrics = spark.table(metrics_ref)
        latest_window = Window.partitionBy("metric_name").orderBy(F.col("metric_time").desc())
        latest = metrics.withColumn("_rn", F.row_number().over(latest_window)).filter(
            F.col("_rn") == 1
        )
        return latest.collect()
    except Exception as exc:
        print(f"metrics table unavailable: {exc}")
        return []


def metric_map(rows):
    return {row["metric_name"]: row.asDict() for row in rows}


def has_alert(metrics, name):
    row = metrics.get(name)
    return bool(row and row.get("status") == "alert")


def metric_value(metrics, name):
    row = metrics.get(name)
    return row.get("metric_value") if row else None


def run_notebook(path, params):
    print(f"RUN {path} params={params}")
    return dbutils.notebook.run(path, notebook_timeout_seconds, params)


def record_action(status, reason, action_type, snapshot):
    action_df = spark.createDataFrame(
        [
            {
                "action_time": datetime.now(timezone.utc),
                "status": status,
                "reason": reason,
                "action_type": action_type,
                "metrics_snapshot": json.dumps(snapshot, default=str),
            }
        ]
    )
    action_df.write.mode("append").saveAsTable(actions_ref)
    display(action_df)


rows = latest_metric_rows()
metrics = metric_map(rows)
snapshot = {
    name: {
        "metric_value": row.get("metric_value"),
        "status": row.get("status"),
        "details": row.get("details"),
    }
    for name, row in metrics.items()
}

alert_names = sorted(name for name, row in metrics.items() if row.get("status") == "alert")
print(f"latest_alert_metrics={alert_names}")

if not alert_names:
    record_action("blocked", "no data quality alerts to remediate", "none", snapshot)
    dbutils.notebook.exit("NO_REMEDIATION_NEEDED")

# COMMAND ----------

safe_actions = []
manual_reasons = []

if has_alert(metrics, "raw_null_open_time_count"):
    manual_reasons.append("raw_null_open_time_count requires landing/parser inspection")

if has_alert(metrics, "raw_duplicate_open_time_count"):
    manual_reasons.append("raw_duplicate_open_time_count requires canonical raw table inspection")

schema_alerts = [name for name in alert_names if name.startswith("schema_drift_")]
if schema_alerts:
    manual_reasons.append(f"schema drift requires manual handling: {schema_alerts}")

if has_alert(metrics, "raw_freshness_hours"):
    safe_actions.append("backfill_raw")

if has_alert(metrics, "raw_features_row_count_delta") or has_alert(
    metrics, "features_target_close_1h_null_count"
):
    safe_actions.append("rebuild_features")

if has_alert(metrics, "prediction_age_hours") or metric_value(metrics, "prediction_count") == 0:
    safe_actions.append("rerun_prediction")

safe_actions = list(dict.fromkeys(safe_actions))

if manual_reasons and not safe_actions:
    record_action("blocked", "; ".join(manual_reasons), "manual_required", snapshot)
    dbutils.notebook.exit("MANUAL_REMEDIATION_REQUIRED")

if not safe_actions:
    record_action(
        "blocked",
        "no auto-remediable data quality alerts found",
        "none",
        snapshot,
    )
    dbutils.notebook.exit("NO_DATA_REMEDIATION_ACTION")

# COMMAND ----------

try:
    if "backfill_raw" in safe_actions:
        run_notebook("00_fetch_binance_to_volume", {"catalog": catalog, "limit": backfill_limit})
        run_notebook("01_data_ingestion", {"catalog": catalog})
        run_notebook("02_feature_engineering", {"catalog": catalog})

    if "rebuild_features" in safe_actions and "backfill_raw" not in safe_actions:
        run_notebook("02_feature_engineering", {"catalog": catalog})

    if "rerun_prediction" in safe_actions:
        run_notebook("05_prediction", {"catalog": catalog})

    try:
        run_notebook("06_monitoring", {"catalog": catalog})
    except Exception as exc:
        print(f"monitoring after remediation still has alerts: {exc}")

    status = "succeeded" if not manual_reasons else "attempted"
    reason_parts = [f"safe_actions={safe_actions}"] + manual_reasons
    record_action(status, "; ".join(reason_parts), "+".join(safe_actions), snapshot)
    dbutils.notebook.exit("DATA_REMEDIATION_COMPLETE")
except Exception as exc:
    record_action("failed", str(exc), "+".join(safe_actions), snapshot)
    raise
