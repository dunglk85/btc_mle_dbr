# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 13 - Select Best Challenger

# COMMAND ----------

import mlflow
from pyspark.sql import functions as F

# COMMAND ----------


def get_widget(name, default):
    try:
        dbutils.widgets.text(name, str(default))
        return dbutils.widgets.get(name)
    except Exception:
        return str(default)


catalog = get_widget("catalog", "btc_dev")
lgbm_training_task_key = get_widget("lgbm_training_task_key", "model_training_reg_lgbm")
xgb_training_task_key = get_widget("xgb_training_task_key", "model_training_reg_xgb")
lgbm_replay_task_key = get_widget("lgbm_replay_task_key", "dataset_replay_reg_lgbm")
xgb_replay_task_key = get_widget("xgb_replay_task_key", "dataset_replay_reg_xgb")

print("RUNNING BEST CHALLENGER SELECTION")
print(f"catalog={catalog}")
print(f"lgbm_training_task_key={lgbm_training_task_key}")
print(f"xgb_training_task_key={xgb_training_task_key}")
print(f"lgbm_replay_task_key={lgbm_replay_task_key}")
print(f"xgb_replay_task_key={xgb_replay_task_key}")

# COMMAND ----------


def read_candidate(label, training_task_key, replay_task_key):
    training_status = dbutils.jobs.taskValues.get(
        taskKey=training_task_key,
        key="training_status",
        default="unknown",
    )
    if training_status != "trained":
        raise ValueError(f"Candidate {label} did not train successfully: status={training_status}")

    replay_status = dbutils.jobs.taskValues.get(
        taskKey=replay_task_key,
        key="dataset_replay_status",
        default="unknown",
    )
    if replay_status != "validated":
        raise ValueError(
            f"Candidate {label} dataset replay did not pass: "
            f"task={replay_task_key}; status={replay_status}"
        )

    run_id = dbutils.jobs.taskValues.get(
        taskKey=training_task_key,
        key="run_id",
        default="",
    )
    if not run_id:
        raise ValueError(f"Candidate {label} missing run_id from {training_task_key}")

    run = mlflow.get_run(run_id)
    metrics = run.data.metrics
    missing_metrics = [name for name in ["rmse", "mae", "directional_accuracy"] if name not in metrics]
    if missing_metrics:
        raise ValueError(f"Candidate {label} run {run_id} missing metrics: {missing_metrics}")

    return {
        "label": label,
        "training_task_key": training_task_key,
        "replay_task_key": replay_task_key,
        "run_id": run_id,
        "model_algo": run.data.params.get("model_algo", label),
        "rmse": float(metrics["rmse"]),
        "mae": float(metrics["mae"]),
        "directional_accuracy": float(metrics["directional_accuracy"]),
        "dataset_replay_status": replay_status,
    }


candidates = [
    read_candidate("lightgbm", lgbm_training_task_key, lgbm_replay_task_key),
    read_candidate("xgboost", xgb_training_task_key, xgb_replay_task_key),
]

# Lower RMSE wins; MAE and directional accuracy are deterministic tie-breakers.
selected = sorted(
    candidates,
    key=lambda row: (row["rmse"], row["mae"], -row["directional_accuracy"]),
)[0]

print(f"selected_model_algo={selected['model_algo']}")
print(f"selected_run_id={selected['run_id']}")
print(f"selected_rmse={selected['rmse']}")
print(f"selected_mae={selected['mae']}")
print(f"selected_directional_accuracy={selected['directional_accuracy']}")

# COMMAND ----------

dbutils.jobs.taskValues.set(key="training_status", value="trained")
dbutils.jobs.taskValues.set(key="run_id", value=selected["run_id"])
dbutils.jobs.taskValues.set(key="task_type", value="regression")
dbutils.jobs.taskValues.set(key="dataset_replay_status", value="validated")
dbutils.jobs.taskValues.set(key="selected_model_algo", value=selected["model_algo"])
dbutils.jobs.taskValues.set(key="selected_training_task_key", value=selected["training_task_key"])
dbutils.jobs.taskValues.set(key="selected_replay_task_key", value=selected["replay_task_key"])

# COMMAND ----------

display(spark.createDataFrame(candidates).orderBy(F.col("rmse"), F.col("mae")))
