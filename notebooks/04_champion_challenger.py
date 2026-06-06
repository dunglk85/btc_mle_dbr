# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 04 - Champion vs Challenger

# COMMAND ----------

# MAGIC %pip install lightgbm xgboost

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import json

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
target_col = "target_return_1h"

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


def load_feature_cols(run_id):
    artifact_path = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path="feature_cols.json",
    )
    with open(artifact_path, encoding="utf-8") as feature_file:
        return json.load(feature_file)


def rmse_on_eval_rows(model_uri, feature_cols, eval_pdf):
    X_test = eval_pdf[feature_cols].astype("float64")
    y_test = eval_pdf[target_col].astype("float64")
    model = mlflow.pyfunc.load_model(model_uri)
    predictions = model.predict(X_test)
    errors = y_test.to_numpy() - predictions
    return float((errors ** 2).mean() ** 0.5)

challenger_run = mlflow.get_run(challenger_run_id)
challenger_rmse = challenger_run.data.metrics.get("rmse")
if challenger_rmse is None:
    raise ValueError(f"Run {challenger_run_id} does not have rmse metric")
challenger_rmse = float(challenger_rmse)
challenger_uri = f"runs:/{challenger_run_id}/model"
training_mode = challenger_run.data.params.get("training_mode", "unknown")
features_table = challenger_run.data.params.get("features_table", f"{catalog}.features.btc_features")
features_table_version = challenger_run.data.params.get("features_table_version")
challenger_data_context = {
    "raw_table_version": challenger_run.data.params.get("raw_table_version"),
    "features_table_version": features_table_version,
    "feature_config_version": challenger_run.data.params.get("feature_config_version"),
    "feature_config_id": challenger_run.data.params.get("feature_config_id"),
}
challenger_feature_cols = load_feature_cols(challenger_run_id)

features_reader = spark.read
if features_table_version is not None:
    features_reader = features_reader.option("versionAsOf", int(features_table_version))
source_pdf = features_reader.table(features_table).orderBy("open_time").toPandas()

# COMMAND ----------

registered = mlflow.register_model(challenger_uri, full_model_name)
client.set_registered_model_alias(full_model_name, "Challenger", registered.version)

promote = False
challenger_fair_rmse = challenger_rmse
champion_fair_rmse = -1.0
challenger_eval_rows = 0
champion_eval_rows = 0
try:
    champion_version = client.get_model_version_by_alias(full_model_name, "Champion")
except Exception as exc:
    print(f"No current Champion alias found: {exc}")
    promote = True
else:
    champion_run = mlflow.get_run(champion_version.run_id)
    champion_rmse = champion_run.data.metrics.get("rmse")
    champion_uri = f"models:/{full_model_name}/{champion_version.version}"
    champion_feature_cols = load_feature_cols(champion_version.run_id)
    eval_feature_cols = sorted(set(challenger_feature_cols) | set(champion_feature_cols))
    missing_eval_cols = [col for col in eval_feature_cols if col not in source_pdf.columns]
    if missing_eval_cols:
        raise ValueError(f"Missing evaluation feature columns in {features_table}: {missing_eval_cols}")
    model_pdf = source_pdf[["open_time", target_col, *eval_feature_cols]].dropna()
    if len(model_pdf) < 100:
        raise ValueError(f"Not enough common evaluation rows after dropna: {len(model_pdf)}")
    split_idx = int(len(model_pdf) * 0.8)
    eval_pdf = model_pdf.iloc[split_idx:]
    challenger_fair_rmse = rmse_on_eval_rows(challenger_uri, challenger_feature_cols, eval_pdf)
    champion_fair_rmse = rmse_on_eval_rows(champion_uri, champion_feature_cols, eval_pdf)
    challenger_eval_rows = len(eval_pdf)
    champion_eval_rows = len(eval_pdf)
    champion_data_context = {
        "raw_table_version": champion_run.data.params.get("raw_table_version"),
        "features_table_version": champion_run.data.params.get("features_table_version"),
        "feature_config_version": champion_run.data.params.get("feature_config_version"),
        "feature_config_id": champion_run.data.params.get("feature_config_id"),
    }
    print(f"champion_version={champion_version.version}")
    print(f"champion_rmse={champion_rmse}")
    print(f"champion_fair_rmse={champion_fair_rmse}")
    print(f"champion_eval_rows={champion_eval_rows}")
    print(f"champion_data_context={champion_data_context}")
    promote = challenger_fair_rmse < champion_fair_rmse

print(f"challenger_run_id={challenger_run_id}")
print(f"challenger_rmse={challenger_rmse}")
print(f"challenger_fair_rmse={challenger_fair_rmse}")
print(f"challenger_eval_rows={challenger_eval_rows}")
print(f"training_mode={training_mode}")
print(f"challenger_data_context={challenger_data_context}")

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
                "challenger_fair_rmse": challenger_fair_rmse,
                "champion_fair_rmse": champion_fair_rmse,
                "challenger_eval_rows": int(challenger_eval_rows),
                "champion_eval_rows": int(champion_eval_rows),
                "training_mode": training_mode,
                "challenger_data_context": str(challenger_data_context),
                "promoted": bool(promote),
            }
        ]
    )
)
