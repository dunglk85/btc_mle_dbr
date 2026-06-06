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
challenger_rmse = challenger_run.data.metrics.get("rmse")
if challenger_rmse is None:
    raise ValueError(f"Run {challenger_run_id} does not have rmse metric")
challenger_rmse = float(challenger_rmse)
challenger_uri = f"runs:/{challenger_run_id}/model"
training_mode = challenger_run.data.params.get("training_mode", "unknown")
challenger_data_context = {
    "raw_table_version": challenger_run.data.params.get("raw_table_version"),
    "features_table_version": challenger_run.data.params.get("features_table_version"),
    "feature_config_version": challenger_run.data.params.get("feature_config_version"),
    "feature_config_id": challenger_run.data.params.get("feature_config_id"),
}

print(f"challenger_run_id={challenger_run_id}")
print(f"challenger_rmse={challenger_rmse}")
print(f"training_mode={training_mode}")
print(f"challenger_data_context={challenger_data_context}")

# COMMAND ----------

registered = mlflow.register_model(challenger_uri, full_model_name)
client.set_registered_model_alias(full_model_name, "Challenger", registered.version)

promote = False
try:
    champion_version = client.get_model_version_by_alias(full_model_name, "Champion")
    champion_run = mlflow.get_run(champion_version.run_id)
    champion_rmse = champion_run.data.metrics.get("rmse")
    champion_data_context = {
        "raw_table_version": champion_run.data.params.get("raw_table_version"),
        "features_table_version": champion_run.data.params.get("features_table_version"),
        "feature_config_version": champion_run.data.params.get("feature_config_version"),
        "feature_config_id": champion_run.data.params.get("feature_config_id"),
    }
    print(f"champion_version={champion_version.version}")
    print(f"champion_rmse={champion_rmse}")
    print(f"champion_data_context={champion_data_context}")
    promote = champion_rmse is None or challenger_rmse < float(champion_rmse)
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
                "challenger_data_context": str(challenger_data_context),
                "promoted": bool(promote),
            }
        ]
    )
)
