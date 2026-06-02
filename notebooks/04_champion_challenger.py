# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 04 - Champion vs Challenger

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient

# COMMAND ----------

catalog = "btc_dev"
model_schema = "models"
model_name = "btc_price_model"
full_model_name = f"{catalog}.{model_schema}.{model_name}"
experiment_name = "/Shared/btc_baseline_training"

print("RUNNING SELF-CONTAINED CHAMPION/CHALLENGER NOTEBOOK")
print(f"full_model_name={full_model_name}")
print(f"experiment_name={experiment_name}")

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{model_schema}")
mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

runs = mlflow.search_runs(
    experiment_names=[experiment_name],
    order_by=["metrics.rmse ASC"],
    max_results=1,
)
if runs.empty:
    raise ValueError(f"No MLflow runs found in {experiment_name}")

best_run = runs.iloc[0]
challenger_run_id = best_run["run_id"]
challenger_rmse = float(best_run["metrics.rmse"])
challenger_uri = f"runs:/{challenger_run_id}/model"

print(f"challenger_run_id={challenger_run_id}")
print(f"challenger_rmse={challenger_rmse}")

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
                "promoted": bool(promote),
            }
        ]
    )
)
