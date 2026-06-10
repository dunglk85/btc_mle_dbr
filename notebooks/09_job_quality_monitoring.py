# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 09 - Job Quality Monitoring

# COMMAND ----------

from datetime import datetime, timezone
import json

import requests

from pyspark.sql import functions as F

# COMMAND ----------


def get_widget(name, default):
    try:
        dbutils.widgets.text(name, str(default))
        return dbutils.widgets.get(name)
    except Exception:
        return str(default)


catalog = get_widget("catalog", "btc_simply")
job_name_filter = get_widget("job_name_filter", "BTC")
lookback_runs = int(get_widget("lookback_runs", "20"))
max_duration_minutes = float(get_widget("max_duration_minutes", "60"))
min_success_rate = float(get_widget("min_success_rate", "0.8"))
fail_on_alert = get_widget("fail_on_alert", "false").lower() == "true"

metrics_ref = f"{catalog}.monitoring.pipeline_metrics"

print("RUNNING JOB QUALITY MONITORING NOTEBOOK")
print(f"metrics_ref={metrics_ref}")
print(f"job_name_filter={job_name_filter}")
print(f"lookback_runs={lookback_runs}")

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.monitoring")
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

ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
workspace_url = ctx.apiUrl().get()
token = ctx.apiToken().get()
headers = {"Authorization": f"Bearer {token}"}

metric_time = datetime.now(timezone.utc)
metrics = []


def append_metric(name, value, status="ok", details=""):
    metrics.append(
        {
            "metric_time": metric_time,
            "metric_name": name,
            "metric_value": float(value) if value is not None else None,
            "status": status,
            "details": details,
        }
    )


def api_get(path, params=None):
    response = requests.get(
        f"{workspace_url}{path}", headers=headers, params=params or {}, timeout=30
    )
    response.raise_for_status()
    return response.json()


def run_url(job_id, run_id):
    if not run_id:
        return None
    return f"{workspace_url}/#job/{job_id}/run/{run_id}"


def job_url(job_id):
    return f"{workspace_url}/#job/{job_id}"


def state_summary(state):
    return {
        "life_cycle_state": state.get("life_cycle_state"),
        "result_state": state.get("result_state"),
        "state_message": state.get("state_message"),
    }


def trace_details(job_id, job_name, run=None, **extra):
    run_id = run.get("run_id") if run else None
    state = run.get("state", {}) if run else {}
    details = {
        "job_id": job_id,
        "job_name": job_name,
        "job_url": job_url(job_id),
        "run_id": run_id,
        "run_url": run_url(job_id, run_id),
        "state": state_summary(state),
    }
    details.update(extra)
    return json.dumps(details, default=str)


def task_trace(task):
    return {
        "task_key": task.get("task_key"),
        "run_id": task.get("run_id"),
        "state": state_summary(task.get("state", {})),
    }


# COMMAND ----------

jobs_payload = api_get("/api/2.1/jobs/list", {"limit": 100})
jobs = jobs_payload.get("jobs", [])
matching_jobs = [job for job in jobs if job_name_filter in job.get("settings", {}).get("name", "")]

append_metric("job_quality_matching_job_count", len(matching_jobs), "ok")

for job in matching_jobs:
    job_id = job["job_id"]
    job_name = job.get("settings", {}).get("name", str(job_id))
    runs_payload = api_get(
        "/api/2.1/jobs/runs/list",
        {
            "job_id": job_id,
            "limit": lookback_runs,
            "expand_tasks": "true",
        },
    )
    runs = runs_payload.get("runs", [])
    if not runs:
        append_metric(
            f"job_quality_run_count_{job_id}",
            0,
            "warn",
            trace_details(job_id, job_name, reason="no recent runs"),
        )
        continue

    terminal_runs = [run for run in runs if run.get("state", {}).get("life_cycle_state") == "TERMINATED"]
    successful_runs = [
        run for run in terminal_runs if run.get("state", {}).get("result_state") == "SUCCESS"
    ]
    failed_runs = [
        run for run in terminal_runs if run.get("state", {}).get("result_state") != "SUCCESS"
    ]
    success_rate = len(successful_runs) / len(terminal_runs) if terminal_runs else None
    status = "ok"
    if success_rate is None:
        status = "warn"
    elif success_rate < min_success_rate:
        status = "alert"
    append_metric(
        f"job_quality_success_rate_{job_id}",
        success_rate,
        status,
        trace_details(
            job_id,
            job_name,
            terminal_runs=len(terminal_runs),
            successful_runs=len(successful_runs),
            failed_runs=len(failed_runs),
            min_success_rate=min_success_rate,
        ),
    )
    append_metric(
        f"job_quality_failed_run_count_{job_id}",
        len(failed_runs),
        "alert" if failed_runs else "ok",
        trace_details(
            job_id,
            job_name,
            failed_run_ids=[run.get("run_id") for run in failed_runs],
            lookback_runs=lookback_runs,
        ),
    )

    latest_run = terminal_runs[0] if terminal_runs else runs[0]
    latest_state = latest_run.get("state", {})
    latest_duration_ms = latest_run.get("execution_duration")
    latest_duration_minutes = (
        latest_duration_ms / 60000 if latest_duration_ms is not None else None
    )
    duration_status = "ok"
    if latest_duration_minutes is None:
        duration_status = "warn"
    elif latest_duration_minutes > max_duration_minutes:
        duration_status = "alert"
    append_metric(
        f"job_quality_latest_duration_minutes_{job_id}",
        latest_duration_minutes,
        duration_status,
        trace_details(
            job_id,
            job_name,
            latest_run,
            max_duration_minutes=max_duration_minutes,
            execution_duration_ms=latest_duration_ms,
        ),
    )

    latest_is_terminal = latest_state.get("life_cycle_state") == "TERMINATED"
    latest_success = 1 if latest_state.get("result_state") == "SUCCESS" else 0
    append_metric(
        f"job_quality_latest_success_{job_id}",
        latest_success if latest_is_terminal else None,
        "ok" if latest_success else ("alert" if latest_is_terminal else "warn"),
        trace_details(job_id, job_name, latest_run),
    )

    task_runs = latest_run.get("tasks", [])
    failed_tasks = [
        task for task in task_runs if task.get("state", {}).get("result_state") != "SUCCESS"
    ]
    append_metric(
        f"job_quality_latest_failed_task_count_{job_id}",
        len(failed_tasks),
        "alert" if failed_tasks else "ok",
        trace_details(
            job_id,
            job_name,
            latest_run,
            failed_tasks=[task_trace(task) for task in failed_tasks],
            task_count=len(task_runs),
        ),
    )

# COMMAND ----------

metrics_df = spark.createDataFrame(metrics)
metrics_df.write.mode("append").saveAsTable(metrics_ref)

alert_count = metrics_df.filter(F.col("status") == "alert").count()
warn_count = metrics_df.filter(F.col("status") == "warn").count()

print(f"job_quality_metrics_written={metrics_df.count()}")
print(f"job_quality_alert_count={alert_count}")
print(f"job_quality_warn_count={warn_count}")

display(metrics_df.orderBy("metric_name"))

if fail_on_alert and alert_count > 0:
    raise ValueError(f"Job quality monitoring produced {alert_count} alert metrics")

if alert_count > 0:
    print(
        "JOB_QUALITY_ALERTS_RECORDED: "
        f"{alert_count} alert metrics written; fail_on_alert=false so job continues"
    )
