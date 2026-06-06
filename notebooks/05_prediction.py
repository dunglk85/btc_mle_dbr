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

features_ref = f"{catalog}.{features_schema}.{features_table}"
config_ref = f"{catalog}.{features_schema}.feature_selection_config"
predictions_ref = f"{catalog}.{predictions_schema}.{predictions_table}"
champion_uri = f"models:/{catalog}.{model_schema}.{model_name}@Champion"
full_model_name = f"{catalog}.{model_schema}.{model_name}"

print("RUNNING SELF-CONTAINED PREDICTION NOTEBOOK")
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
        model_uri STRING
    )
    USING DELTA
""")

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
print(f"champion_version={champion_version.version}")
print(f"champion_run_id={champion_run_id}")

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
        config_row = spark.table(config_ref).collect()
        if config_row:
            cols = json.loads(config_row[0]["config_value"])
            print(f"Loaded {len(cols)} fallback selected features from {config_ref}")
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
        }
    ]
).withColumn("prediction_time", F.current_timestamp())

pred_df.select(
    "prediction_time",
    "feature_open_time",
    "predicted_close",
    "model_uri",
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
