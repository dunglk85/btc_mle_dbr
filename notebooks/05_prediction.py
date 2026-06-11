# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 05 - Prediction

# COMMAND ----------

# MAGIC %pip install lightgbm xgboost scikit-learn

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import mlflow
import json
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql import types as T
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

# COMMAND ----------

def get_widget(name, default):
    try:
        dbutils.widgets.text(name, str(default))
        return dbutils.widgets.get(name)
    except Exception:
        return str(default)


catalog = get_widget("catalog", "btc_simply")
features_schema = "features"
predictions_schema = "predictions"
features_table = "btc_features"
predictions_table = "btc_predictions"
model_schema = "models"
model_name = "btc_price_model"
allow_default_feature_fallback = get_widget("allow_default_feature_fallback", "false").lower() == "true"

raw_ref = f"{catalog}.raw.btc_hourly"
features_ref = f"{catalog}.{features_schema}.{features_table}"
config_ref = f"{catalog}.{features_schema}.feature_selection_config"
predictions_ref = f"{catalog}.{predictions_schema}.{predictions_table}"
champion_uri = f"models:/{catalog}.{model_schema}.{model_name}@Champion"
full_model_name = f"{catalog}.{model_schema}.{model_name}"
training_manifest_ref = f"{catalog}.monitoring.training_dataset_manifests"
champion_status_ref = f"{catalog}.monitoring.champion_model_status"

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
        predicted_return_1h DOUBLE,
        model_uri STRING,
        model_version STRING,
        model_run_id STRING,
        model_target_col STRING,
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
    "predicted_return_1h DOUBLE",
    "model_version STRING",
    "model_run_id STRING",
    "model_target_col STRING",
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


def load_champion_target_col(run_id):
    target_col = champion_run.data.params.get("target_col")
    if target_col:
        return target_col

    try:
        artifact_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path="training_dataset_manifest.json",
        )
        with open(artifact_path, encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)
        target_col = manifest.get("target_col")
        if target_col:
            return target_col
    except Exception as exc:
        print(f"WARNING: Could not load Champion training target metadata ({exc})")

    print("WARNING: Champion target metadata missing; assuming target_return_1h")
    return "target_return_1h"


def load_feature_cols_for_champion(run_id, run_params):
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
        manifest_rows = (
            spark.table(training_manifest_ref)
            .filter(F.col("run_id") == run_id)
            .orderBy(F.col("created_at").desc())
            .limit(1)
            .collect()
        )
        if manifest_rows:
            cols = json.loads(manifest_rows[0]["feature_cols_json"])
            print(f"Loaded {len(cols)} features from Champion training manifest")
            return cols
    except Exception as manifest_exc:
        print(f"WARNING: Could not load Champion manifest feature columns ({manifest_exc})")

    config_table = run_params.get("feature_config_table")
    config_version = run_params.get("feature_config_version")
    config_id = run_params.get("feature_config_id")
    if config_table and config_version is not None:
        try:
            config = spark.read.option("versionAsOf", int(config_version)).table(config_table)
            config = config.filter(F.col("config_key") == "selected_features")
            if config_id is not None and int(config_id) >= 0:
                if "config_id" in config.columns:
                    config = config.filter(F.col("config_id") == int(config_id))
                else:
                    config = config.filter(F.col("config_version") == int(config_id))
            config_row = config.orderBy(F.col("created_at").desc()).limit(1).collect()
            if config_row:
                cols = json.loads(config_row[0]["config_value"])
                print(
                    f"Loaded {len(cols)} features from Champion versioned feature config "
                    f"config_version={config_version}; config_id={config_id}"
                )
                return cols
        except Exception as config_version_exc:
            print(f"WARNING: Could not load Champion versioned feature config ({config_version_exc})")

    if catalog != "btc_simply" and not allow_default_feature_fallback:
        raise ValueError(
            "Could not resolve Champion feature columns from artifact, manifest, or versioned config. "
            f"Refusing active-config fallback outside btc_simply; catalog={catalog}"
        )

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
                f"Loaded {len(cols)} fallback ACTIVE selected features from {config_ref} "
                f"config_version={config_id}"
            )
            return cols
        raise ValueError("Empty feature selection config table")
    except Exception as config_exc:
        if allow_default_feature_fallback:
            print(f"WARNING: Using default feature fallback because active config could not be loaded: {config_exc}")
            return [
                "return_1h", "return_6h", "return_24h",
                "close_ma7_ratio", "close_ma24_ratio", "close_ma168_ratio",
                "macd", "macd_signal", "macd_hist",
                "rsi_14", "atr_14", "atr_ratio", "bb_width",
                "volume_ratio", "log_volume", "hl_spread", "oc_change",
                "close_lag_1h", "close_lag_2h", "close_lag_4h", "close_lag_12h", "close_lag_24h",
                "hour", "day_of_week", "hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
                "volatility_12h", "volatility_24h", "volatility_168h",
                "volatility_zscore",
                "roc_3h", "roc_6h", "roc_12h",
                "price_acceleration",
                "vol_price_divergence",
                "return_skew_168h", "return_kurt_168h",
                "vol_ratio_12_168",
            ]
        raise ValueError(
            "Could not resolve prediction feature columns from Champion artifact or feature config"
        ) from config_exc


model_target_col = load_champion_target_col(champion_run_id)
feature_cols = load_feature_cols_for_champion(champion_run_id, champion_run.data.params)
print(f"model_target_col={model_target_col}")

# COMMAND ----------

# Publish current Champion metadata for dashboards and operators.
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.monitoring")
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {champion_status_ref} (
        updated_at TIMESTAMP,
        model_name STRING,
        model_alias STRING,
        model_version STRING,
        model_run_id STRING,
        model_algo STRING,
        model_target_col STRING,
        rmse DOUBLE,
        mae DOUBLE,
        r2 DOUBLE,
        mape DOUBLE,
        directional_accuracy DOUBLE,
        raw_table_version BIGINT,
        features_table_version BIGINT,
        feature_config_version BIGINT,
        feature_config_id BIGINT,
        n_features BIGINT,
        feature_cols_json STRING
    )
    USING DELTA
""")

def metric_or_none(name):
    value = champion_run.data.metrics.get(name)
    return float(value) if value is not None else None


champion_status_schema = T.StructType([
    T.StructField("model_name", T.StringType(), True),
    T.StructField("model_alias", T.StringType(), True),
    T.StructField("model_version", T.StringType(), True),
    T.StructField("model_run_id", T.StringType(), True),
    T.StructField("model_algo", T.StringType(), True),
    T.StructField("model_target_col", T.StringType(), True),
    T.StructField("rmse", T.DoubleType(), True),
    T.StructField("mae", T.DoubleType(), True),
    T.StructField("r2", T.DoubleType(), True),
    T.StructField("mape", T.DoubleType(), True),
    T.StructField("directional_accuracy", T.DoubleType(), True),
    T.StructField("raw_table_version", T.LongType(), True),
    T.StructField("features_table_version", T.LongType(), True),
    T.StructField("feature_config_version", T.LongType(), True),
    T.StructField("feature_config_id", T.LongType(), True),
    T.StructField("n_features", T.LongType(), True),
    T.StructField("feature_cols_json", T.StringType(), True),
])

champion_status_df = spark.createDataFrame(
    [
        {
            "model_name": full_model_name,
            "model_alias": "Champion",
            "model_version": str(champion_version.version),
            "model_run_id": champion_run_id,
            "model_algo": champion_run.data.params.get("model_algo"),
            "model_target_col": model_target_col,
            "rmse": metric_or_none("rmse"),
            "mae": metric_or_none("mae"),
            "r2": metric_or_none("r2"),
            "mape": metric_or_none("mape"),
            "directional_accuracy": metric_or_none("directional_accuracy"),
            "raw_table_version": model_raw_table_version,
            "features_table_version": model_features_table_version,
            "feature_config_version": model_feature_config_version,
            "feature_config_id": model_feature_config_id,
            "n_features": len(feature_cols),
            "feature_cols_json": json.dumps(feature_cols),
        }
    ],
    champion_status_schema,
).withColumn("updated_at", F.current_timestamp())

champion_status_df.select(
    "updated_at",
    "model_name",
    "model_alias",
    "model_version",
    "model_run_id",
    "model_algo",
    "model_target_col",
    "rmse",
    "mae",
    "r2",
    "mape",
    "directional_accuracy",
    "raw_table_version",
    "features_table_version",
    "feature_config_version",
    "feature_config_id",
    "n_features",
    "feature_cols_json",
).createOrReplaceTempView("_champion_model_status")

spark.sql(f"""
    MERGE INTO {champion_status_ref} AS target
    USING _champion_model_status AS source
    ON target.model_name = source.model_name
       AND target.model_alias = source.model_alias
    WHEN MATCHED THEN UPDATE SET *
    WHEN NOT MATCHED THEN INSERT *
""")
print(f"champion_status_written={champion_status_ref}")

source = spark.table(features_ref)
missing_features = [col for col in feature_cols if col not in source.columns]
if missing_features:
    raise ValueError(f"Missing prediction features in {features_ref}: {missing_features}")
if "close" not in source.columns:
    raise ValueError(f"Missing close column in {features_ref}; cannot convert return prediction to close")

select_cols = ["open_time", *feature_cols]
if "close" not in feature_cols:
    select_cols.append("close")

latest = source.select(*select_cols).dropna(subset=feature_cols + ["close"]).orderBy(
    F.col("open_time").desc()
).limit(1)
latest_rows = latest.collect()
if not latest_rows:
    print("no_feature_rows=skip")
    dbutils.notebook.exit("SKIPPED: No feature rows available for prediction")

feature_open_time = latest_rows[0]["open_time"]
latest_close = float(latest_rows[0]["close"])
latest_pdf = latest.select(*feature_cols).toPandas().astype("float64")

# COMMAND ----------

try:
    champion = mlflow.pyfunc.load_model(champion_uri)
except MlflowException as exc:
    print(f"SKIP_PREDICTION_NO_CHAMPION: {champion_uri}")
    print(f"mlflow_error={exc}")
    dbutils.notebook.exit("SKIP_PREDICTION_NO_CHAMPION")

raw_prediction = float(champion.predict(pd.DataFrame(latest_pdf, columns=feature_cols))[0])
if model_target_col == "target_return_1h":
    predicted_return_1h = raw_prediction
    predicted_close = latest_close * (1.0 + predicted_return_1h)
elif model_target_col == "target_close_1h":
    predicted_close = raw_prediction
    predicted_return_1h = (predicted_close / latest_close) - 1.0
else:
    raise ValueError(f"Unsupported Champion target column for prediction: {model_target_col}")

print(f"feature_open_time={feature_open_time}")
print(f"latest_close={latest_close:.4f}")
print(f"predicted_return_1h={predicted_return_1h:.8f}")
print(f"predicted_close={predicted_close:.4f}")

# COMMAND ----------

pred_df = spark.createDataFrame(
    [
        {
            "feature_open_time": feature_open_time,
            "predicted_close": float(predicted_close),
            "predicted_return_1h": float(predicted_return_1h),
            "model_uri": champion_uri,
            "model_version": str(champion_version.version),
            "model_run_id": champion_run_id,
            "model_target_col": model_target_col,
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
    "predicted_return_1h",
    "model_uri",
    "model_version",
    "model_run_id",
    "model_target_col",
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
       AND target.model_version = source.model_version
       AND target.model_run_id = source.model_run_id
    WHEN MATCHED THEN UPDATE SET *
    WHEN NOT MATCHED THEN INSERT *
""")

display(pred_df)
