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
model_schema = "models"
model_name = "btc_price_model"
full_model_name = f"{catalog}.{model_schema}.{model_name}"
experiment_name = "/Shared/btc_baseline_training"

print("RUNNING SELF-CONTAINED CHAMPION/CHALLENGER NOTEBOOK")
print(f"full_model_name={full_model_name}")
print(f"experiment_name={experiment_name}")

training_status = dbutils.jobs.taskValues.get(
    taskKey="model_training",
    key="training_status",
    default="unknown",
)
if training_status != "trained":
    print(f"SKIP_REGISTRY: training_status={training_status}")
    dbutils.notebook.exit("SKIP_REGISTRY")

challenger_run_id = dbutils.jobs.taskValues.get(
    taskKey="model_training",
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

print(f"challenger_run_id={challenger_run_id}")
print(f"challenger_rmse={challenger_rmse}")
print(f"training_mode={training_mode}")

# COMMAND ----------

registered = mlflow.register_model(challenger_uri, full_model_name)
client.set_registered_model_alias(full_model_name, "Challenger", registered.version)

promote = False
try:
    champion_version = client.get_model_version_by_alias(full_model_name, "Champion")
    champion_run = mlflow.get_run(champion_version.run_id)
    champion_rmse = champion_run.data.metrics.get("rmse")
    print(f"champion_version={champion_version.version}")
    print(f"champion_rmse={champion_rmse}")
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
                "promoted": bool(promote),
            }
        ]
    )
)
