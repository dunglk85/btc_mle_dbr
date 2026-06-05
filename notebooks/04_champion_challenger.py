# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 04 - Champion vs Challenger

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient

# COMMAND ----------

def get_widget(name, default):
    try:
        dbutils.widgets.text(name, str(default))
        return dbutils.widgets.get(name)
    except Exception:
        return str(default)


catalog = get_widget("catalog", "btc_dev")
# Define the task type (regression or classification) for metric comparison
task_type = get_widget("task_type", "regression")
training_task_key = get_widget("training_task_key", "model_training")
model_schema = "models"
model_name = "btc_price_model"
full_model_name = f"{catalog}.{model_schema}.{model_name}"
experiment_name = "/Shared/btc_baseline_training"

print("RUNNING SELF-CONTAINED CHAMPION/CHALLENGER NOTEBOOK")
print(f"full_model_name={full_model_name}")
print(f"experiment_name={experiment_name}")

training_status = dbutils.jobs.taskValues.get(
    taskKey=training_task_key,
    key="training_status",
    default="unknown",
)
if training_status != "trained":
    print(f"SKIP_REGISTRY: training_status={training_status}")
    dbutils.notebook.exit("SKIP_REGISTRY")

challenger_run_id = dbutils.jobs.taskValues.get(
    taskKey=training_task_key,
    key="run_id",
    default="",
)
if not challenger_run_id:
    raise ValueError("Missing run_id task value from model_training")

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{model_schema}")
mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

challenger_run = mlflow.get_run(challenger_run_id)
# Choose metric based on task_type
metric_name = "rmse" if task_type == "regression" else "f1_score"
challenger_metric = challenger_run.data.metrics.get(metric_name)
if challenger_metric is None:
    raise ValueError(f"Run {challenger_run_id} does not have {metric_name} metric")
challenger_metric = float(challenger_metric)
challenger_uri = f"runs:/{challenger_run_id}/model"
# Preserve original variable name for downstream logic
if task_type == "regression":
    challenger_rmse = challenger_metric
else:
    challenger_f1 = challenger_metric
training_mode = challenger_run.data.params.get("training_mode", "unknown")

print(f"challenger_run_id={challenger_run_id}")
if task_type == "regression":
    print(f"challenger_rmse={challenger_rmse}")
else:
    print(f"challenger_f1={challenger_f1}")
print(f"training_mode={training_mode}")

# COMMAND ----------

registered = mlflow.register_model(challenger_uri, full_model_name)
client.set_registered_model_alias(full_model_name, "Challenger", registered.version)

promote = False
try:
    champion_version = client.get_model_version_by_alias(full_model_name, "Champion")
    champion_run = mlflow.get_run(champion_version.run_id)
    # Retrieve champion metric according to task_type
    if task_type == "regression":
        champion_metric = champion_run.data.metrics.get("rmse")
        print(f"champion_version={champion_version.version}")
        print(f"champion_rmse={champion_metric}")
        promote = champion_metric is None or challenger_rmse < float(champion_metric)
    else:
        champion_metric = champion_run.data.metrics.get("f1_score")
        print(f"champion_version={champion_version.version}")
        print(f"champion_f1={champion_metric}")
        promote = champion_metric is None or challenger_f1 > float(champion_metric)
except Exception as exc:
    print(f"No current Champion alias found: {exc}")
    promote = True

if promote:
    client.set_registered_model_alias(full_model_name, "Champion", registered.version)
    print(f"PROMOTED Challenger version {registered.version} to Champion")
else:
    print("Champion retained")

# COMMAND ----------

display(
    spark.createDataFrame(
        [
            {
                "registered_version": str(registered.version),
                "challenger_run_id": challenger_run_id,
                "challenger_rmse": challenger_rmse,
                "training_mode": training_mode,
                "promoted": bool(promote),
            }
        ]
    )
)
