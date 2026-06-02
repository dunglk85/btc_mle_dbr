# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 05 - Prediction

# COMMAND ----------

import mlflow
import pandas as pd
from pyspark.sql import functions as F
from mlflow.exceptions import MlflowException

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
predictions_ref = f"{catalog}.{predictions_schema}.{predictions_table}"
champion_uri = f"models:/{catalog}.{model_schema}.{model_name}@Champion"

print("RUNNING SELF-CONTAINED PREDICTION NOTEBOOK")
print(f"features_ref={features_ref}")
print(f"predictions_ref={predictions_ref}")
print(f"champion_uri={champion_uri}")

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

feature_cols = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "trades",
    "ma_7",
    "ma_24",
    "ma_168",
    "close_lag_1h",
    "close_lag_2h",
    "close_lag_4h",
    "close_lag_12h",
    "close_lag_24h",
    "return_1h",
    "hl_spread",
    "oc_change",
    "hour",
    "day_of_week",
]

latest = spark.table(features_ref).select("open_time", *feature_cols).dropna().orderBy(
    F.col("open_time").desc()
).limit(1)
latest_rows = latest.collect()
if not latest_rows:
    raise ValueError(f"No feature rows available in {features_ref}")

feature_open_time = latest_rows[0]["open_time"]
latest_pdf = latest.select(*feature_cols).toPandas()

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
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
