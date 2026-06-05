# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 11 - Trigger Model Refresh
# MAGIC
# MAGIC Trigger `btc_model_refresh_job` only when the latest training gate decision allows retraining.

# COMMAND ----------

from datetime import datetime, timezone

# COMMAND ----------

def get_widget(name, default):
    try:
        dbutils.widgets.text(name, str(default))
        return dbutils.widgets.get(name)
    except Exception:
        return str(default)


catalog = get_widget("catalog", "btc_dev")
model_refresh_job_id = get_widget("model_refresh_job_id", "")
model_refresh_job_name = get_widget("model_refresh_job_name", "")
max_decision_age_hours = float(get_widget("max_decision_age_hours", "12"))
expected_trigger_mode = get_widget("expected_trigger_mode", "drift")

if not model_refresh_job_id and not model_refresh_job_name:
    raise ValueError("model_refresh_job_id or model_refresh_job_name is required")

decisions_ref = f"{catalog}.monitoring.model_refresh_decisions"

print("RUNNING MODEL REFRESH TRIGGER NOTEBOOK")
print(f"decisions_ref={decisions_ref}")
print(f"model_refresh_job_id={model_refresh_job_id}")
print(f"model_refresh_job_name={model_refresh_job_name}")
print(f"max_decision_age_hours={max_decision_age_hours}")
print(f"expected_trigger_mode={expected_trigger_mode}")

# COMMAND ----------

latest_decision = (
    spark.table(decisions_ref)
    .orderBy("decision_time", ascending=False)
    .limit(1)
    .collect()
)

if not latest_decision:
    print("No training gate decision found")
    dbutils.notebook.exit("SKIP_MODEL_REFRESH_NO_DECISION")

decision = latest_decision[0]
decision_time = decision["decision_time"]
decision_time_utc = decision_time.replace(tzinfo=timezone.utc)
decision_age_hours = (datetime.now(timezone.utc) - decision_time_utc).total_seconds() / 3600

print(f"decision_time={decision_time}")
print(f"decision_age_hours={decision_age_hours:.2f}")
print(f"should_retrain={decision['should_retrain']}")
print(f"trigger_mode={decision['trigger_mode']}")
print(f"reason={decision['reason']}")

if decision_age_hours > max_decision_age_hours:
    dbutils.notebook.exit(
        f"SKIP_MODEL_REFRESH_STALE_DECISION: age_hours={decision_age_hours:.2f}; reason={decision['reason']}"
    )

if decision["trigger_mode"] != expected_trigger_mode:
    dbutils.notebook.exit(
        "SKIP_MODEL_REFRESH_TRIGGER_MODE_MISMATCH: "
        f"actual={decision['trigger_mode']}; expected={expected_trigger_mode}; reason={decision['reason']}"
    )

if not decision["should_retrain"]:
    if "blocking" in decision["reason"] or "stale" in decision["reason"] or "missing" in decision["reason"]:
        skip_status = "SKIP_MODEL_REFRESH_BLOCKED"
    else:
        skip_status = "SKIP_MODEL_REFRESH_NO_RETRAIN_TRIGGER"
    dbutils.notebook.exit(f"{skip_status}: reason={decision['reason']}")

# COMMAND ----------

try:
    from databricks.sdk import WorkspaceClient
except ImportError as exc:
    raise RuntimeError("databricks-sdk is required to trigger model refresh job") from exc

workspace = WorkspaceClient()
if model_refresh_job_id:
    job_id = int(model_refresh_job_id)
else:
    matches = list(workspace.jobs.list(name=model_refresh_job_name))
    if not matches:
        raise ValueError(f"Could not find Databricks job named {model_refresh_job_name}")
    if len(matches) > 1:
        raise ValueError(f"Found multiple Databricks jobs named {model_refresh_job_name}")
    job_id = matches[0].job_id

run = workspace.jobs.run_now(job_id=job_id)
print(f"triggered_job_id={job_id}")
print(f"triggered_run_id={run.run_id}")

dbutils.notebook.exit(f"TRIGGERED_MODEL_REFRESH_RUN_ID={run.run_id}")
