# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 05 - Prediction

# COMMAND ----------

# MAGIC %pip install lightgbm xgboost

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import mlflow
import json
import pandas as pd
from pyspark.sql import functions as F
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

# COMMAND ----------

def get_widget(name, default):
    try:
        dbutils.widgets.text(name, str(default))
        return dbutils.widgets.get(name)
    except Exception:
        return str(default)


catalog = get_widget("catalog", "btc_dev")
features_schema = "features"
predictions_schema = "predictions"
features_table = "btc_features"
predictions_table = "btc_predictions"
model_schema = "models"
model_name = "btc_price_model"

raw_ref = f"{catalog}.raw.btc_hourly"
features_ref = f"{catalog}.{features_schema}.{features_table}"
config_ref = f"{catalog}.{features_schema}.feature_selection_config"
predictions_ref = f"{catalog}.{predictions_schema}.{predictions_table}"
champion_uri = f"models:/{catalog}.{model_schema}.{model_name}@Champion"
full_model_name = f"{catalog}.{model_schema}.{model_name}"

print("RUNNING SELF-CONTAINED PREDICTION NOTEBOOK")
print(f"raw_ref={raw_ref}")
print(f"features_ref={features_ref}")
print(f"config_ref={config_ref}")
print(f"predictions_ref={predictions_ref}")
print(f"champion_uri={champion_uri}")
print(f"full_model_name={full_model_name}")

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{predictions_schema}")
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {predictions_ref} (
        prediction_time TIMESTAMP,
        feature_open_time TIMESTAMP,
        predicted_close DOUBLE,
        model_uri STRING,
        model_version STRING,
        model_run_id STRING,
        raw_table_version BIGINT,
        features_table_version BIGINT,
        model_raw_table_version BIGINT,
        model_features_table_version BIGINT,
        model_feature_config_version BIGINT,
        model_feature_config_id BIGINT
    )
    USING DELTA
""")

for column_def in [
    "model_version STRING",
    "model_run_id STRING",
    "raw_table_version BIGINT",
    "features_table_version BIGINT",
    "model_raw_table_version BIGINT",
    "model_features_table_version BIGINT",
    "model_feature_config_version BIGINT",
    "model_feature_config_id BIGINT",
]:
    try:
        spark.sql(f"ALTER TABLE {predictions_ref} ADD COLUMNS ({column_def})")
    except Exception as exc:
        print(f"Column already exists or could not be added ({column_def}): {exc}")


def latest_delta_version(table_ref):
    history = spark.sql(f"DESCRIBE HISTORY {table_ref} LIMIT 1").collect()
    if not history:
        raise ValueError(f"No Delta history found for {table_ref}")
    return int(history[0]["version"])


raw_table_version = latest_delta_version(raw_ref)
features_table_version = latest_delta_version(features_ref)
print(f"raw_table_version={raw_table_version}")
print(f"features_table_version={features_table_version}")

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

try:
    champion_version = client.get_model_version_by_alias(full_model_name, "Champion")
except Exception as exc:
    print(f"SKIP_PREDICTION_NO_CHAMPION: {champion_uri}")
    print(f"mlflow_error={exc}")
    dbutils.notebook.exit("SKIP_PREDICTION_NO_CHAMPION")

champion_run_id = champion_version.run_id
champion_run = mlflow.get_run(champion_run_id)
model_raw_table_version = int(champion_run.data.params.get("raw_table_version", -1))
model_features_table_version = int(champion_run.data.params.get("features_table_version", -1))
model_feature_config_version = int(champion_run.data.params.get("feature_config_version", -1))
model_feature_config_id = int(champion_run.data.params.get("feature_config_id", -1))
print(f"champion_version={champion_version.version}")
print(f"champion_run_id={champion_run_id}")
print(f"model_raw_table_version={model_raw_table_version}")
print(f"model_features_table_version={model_features_table_version}")
print(f"model_feature_config_version={model_feature_config_version}")
print(f"model_feature_config_id={model_feature_config_id}")

def load_feature_cols_for_champion(run_id):
    try:
        artifact_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path="feature_cols.json",
        )
        with open(artifact_path, encoding="utf-8") as feature_file:
            cols = json.load(feature_file)
        print(f"Loaded {len(cols)} features from Champion artifact feature_cols.json")
        return cols
    except Exception as artifact_exc:
        print(f"WARNING: Could not load Champion feature artifact ({artifact_exc})")

    try:
        config_row = (
            spark.table(config_ref)
            .filter(F.col("config_key") == "selected_features")
            .filter(F.col("is_active") == True)
            .orderBy(F.col("created_at").desc())
            .limit(1)
            .collect()
        )
        if config_row:
            cols = json.loads(config_row[0]["config_value"])
            config_dict = config_row[0].asDict()
            config_id = config_dict.get("config_id") or config_dict.get("config_version")
            print(
                f"Loaded {len(cols)} fallback selected features from {config_ref} "
                f"config_version={config_id}"
            )
            return cols
        raise ValueError("Empty feature selection config table")
    except Exception as config_exc:
        raise ValueError(
            "Could not resolve prediction feature columns from Champion artifact or feature config"
        ) from config_exc


feature_cols = load_feature_cols_for_champion(champion_run_id)

source = spark.table(features_ref)
missing_features = [col for col in feature_cols if col not in source.columns]
if missing_features:
    raise ValueError(f"Missing prediction features in {features_ref}: {missing_features}")

latest = source.select("open_time", *feature_cols).dropna().orderBy(
    F.col("open_time").desc()
).limit(1)
latest_rows = latest.collect()
if not latest_rows:
    raise ValueError(f"No feature rows available in {features_ref}")

feature_open_time = latest_rows[0]["open_time"]
latest_pdf = latest.select(*feature_cols).toPandas().astype("float64")

# COMMAND ----------

try:
    champion = mlflow.pyfunc.load_model(champion_uri)
except MlflowException as exc:
    print(f"SKIP_PREDICTION_NO_CHAMPION: {champion_uri}")
    print(f"mlflow_error={exc}")
    dbutils.notebook.exit("SKIP_PREDICTION_NO_CHAMPION")

prediction = float(champion.predict(pd.DataFrame(latest_pdf, columns=feature_cols))[0])
print(f"feature_open_time={feature_open_time}")
print(f"predicted_close={prediction:.4f}")

# COMMAND ----------

pred_df = spark.createDataFrame(
    [
        {
            "feature_open_time": feature_open_time,
            "predicted_close": prediction,
            "model_uri": champion_uri,
            "model_version": str(champion_version.version),
            "model_run_id": champion_run_id,
            "raw_table_version": raw_table_version,
            "features_table_version": features_table_version,
            "model_raw_table_version": model_raw_table_version,
            "model_features_table_version": model_features_table_version,
            "model_feature_config_version": model_feature_config_version,
            "model_feature_config_id": model_feature_config_id,
        }
    ]
).withColumn("prediction_time", F.current_timestamp())

pred_df.select(
    "prediction_time",
    "feature_open_time",
    "predicted_close",
    "model_uri",
    "model_version",
    "model_run_id",
    "raw_table_version",
    "features_table_version",
    "model_raw_table_version",
    "model_features_table_version",
    "model_feature_config_version",
    "model_feature_config_id",
).createOrReplaceTempView("_btc_prediction")

spark.sql(f"""
    MERGE INTO {predictions_ref} AS target
    USING _btc_prediction AS source
    ON target.feature_open_time = source.feature_open_time
       AND target.model_uri = source.model_uri
    WHEN MATCHED THEN UPDATE SET *
    WHEN NOT MATCHED THEN INSERT *
""")

display(pred_df)
