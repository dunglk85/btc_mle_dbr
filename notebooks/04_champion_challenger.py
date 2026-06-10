# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 04 - Champion vs Challenger

# COMMAND ----------

# MAGIC %pip install lightgbm xgboost scikit-learn

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import json

import mlflow
import numpy as np
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from pyspark.sql import functions as F
from pyspark.sql import types as T

# COMMAND ----------

def get_widget(name, default):
    try:
        dbutils.widgets.text(name, str(default))
        return dbutils.widgets.get(name)
    except Exception:
        return str(default)


catalog = get_widget("catalog", "btc_simply")
lgbm_training_task_key = get_widget("lgbm_training_task_key", "model_training_reg_lgbm")
xgb_training_task_key = get_widget("xgb_training_task_key", "model_training_reg_xgb")
rf_training_task_key = get_widget("rf_training_task_key", "model_training_reg_rf")
max_evaluation_rows = int(get_widget("max_evaluation_rows", "2000"))
min_evaluation_rows = int(get_widget("min_evaluation_rows", "100"))
model_schema = "models"
model_name = "btc_price_model"
full_model_name = f"{catalog}.{model_schema}.{model_name}"
experiment_name = "/Shared/btc_baseline_training"
target_col = "target_return_1h"
training_manifest_ref = f"{catalog}.monitoring.training_dataset_manifests"

print("RUNNING SELF-CONTAINED CHAMPION/CHALLENGER NOTEBOOK")
print(f"full_model_name={full_model_name}")
print(f"experiment_name={experiment_name}")
print(f"lgbm_training_task_key={lgbm_training_task_key}")
print(f"xgb_training_task_key={xgb_training_task_key}")
print(f"rf_training_task_key={rf_training_task_key}")
print(f"max_evaluation_rows={max_evaluation_rows}")

def read_candidate(label, training_task_key):
    training_status = dbutils.jobs.taskValues.get(
        taskKey=training_task_key,
        key="training_status",
        default="unknown",
    )
    if training_status != "trained":
        raise ValueError(f"Candidate {label} did not train successfully: status={training_status}")

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
        "run_id": run_id,
        "model_algo": run.data.params.get("model_algo", label),
        "rmse": float(metrics["rmse"]),
        "mae": float(metrics["mae"]),
        "directional_accuracy": float(metrics["directional_accuracy"]),
    }


candidates = [
    read_candidate("lightgbm", lgbm_training_task_key),
    read_candidate("xgboost", xgb_training_task_key),
    read_candidate("random_forest", rf_training_task_key),
]
selected_candidate = sorted(
    candidates,
    key=lambda row: (row["rmse"], row["mae"], -row["directional_accuracy"]),
)[0]

challenger_run_id = selected_candidate["run_id"]

print(f"selected_model_algo={selected_candidate['model_algo']}")
print(f"selected_run_id={challenger_run_id}")
print(f"selected_rmse={selected_candidate['rmse']}")
print(f"selected_mae={selected_candidate['mae']}")
print(f"selected_directional_accuracy={selected_candidate['directional_accuracy']}")

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{model_schema}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.monitoring")
mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()


def tag_feature_cols_source(run_id, label, source):
    client.set_tag(run_id, "feature_cols_source", source)
    client.set_tag(challenger_run_id, f"{label}_feature_cols_source", source)


def load_feature_cols(run_id, run_params, label):
    try:
        artifact_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path="feature_cols.json",
        )
        with open(artifact_path, encoding="utf-8") as feature_file:
            cols = json.load(feature_file)
        print(f"Loaded {len(cols)} feature columns from MLflow artifact for run_id={run_id}")
        tag_feature_cols_source(run_id, label, "mlflow_artifact:feature_cols.json")
        return cols
    except FileNotFoundError as artifact_exc:
        print(
            f"WARNING: feature_cols.json artifact not found for run_id={run_id}; "
            f"falling back to manifest/config: {artifact_exc}"
        )
    except MlflowException as artifact_exc:
        if "RESOURCE_DOES_NOT_EXIST" not in str(artifact_exc):
            raise
        print(
            f"WARNING: feature_cols.json artifact not found for run_id={run_id}; "
            f"falling back to manifest/config: {artifact_exc}"
        )

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
            print(f"Loaded {len(cols)} feature columns from manifest for run_id={run_id}")
            tag_feature_cols_source(run_id, label, "delta_manifest")
            return cols
    except Exception as manifest_exc:
        raise RuntimeError(
            f"Failed to load manifest feature columns for run_id={run_id}"
        ) from manifest_exc

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
            config_rows = config.orderBy(F.col("created_at").desc()).limit(1).collect()
            if config_rows:
                cols = json.loads(config_rows[0]["config_value"])
                print(f"Loaded {len(cols)} feature columns from feature config for run_id={run_id}")
                tag_feature_cols_source(run_id, label, "feature_config")
                return cols
        except Exception as config_exc:
            raise RuntimeError(
                f"Failed to load config feature columns for run_id={run_id}"
            ) from config_exc

    raise ValueError(f"Could not resolve feature columns for run_id={run_id}")


def evaluate_on_rows(model_uri, feature_cols, eval_pdf):
    X_test = eval_pdf[feature_cols].astype("float64")
    y_test = eval_pdf[target_col].astype("float64")
    model = mlflow.pyfunc.load_model(model_uri)
    predictions = np.asarray(model.predict(X_test), dtype="float64")
    actuals = y_test.to_numpy()
    errors = actuals - predictions
    return {
        "rmse": float((errors ** 2).mean() ** 0.5),
        "mae": float(np.abs(errors).mean()),
        "directional_accuracy": float((np.sign(actuals) == np.sign(predictions)).mean()),
    }

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
challenger_feature_cols = load_feature_cols(
    challenger_run_id,
    challenger_run.data.params,
    "challenger",
)

# COMMAND ----------

promote = False
registered = None
challenger_fair_rmse = challenger_rmse
challenger_fair_mae = None
challenger_directional_accuracy = None
champion_fair_rmse = None
champion_fair_mae = None
champion_directional_accuracy = None
challenger_eval_rows = 0
champion_eval_rows = 0
champion_version = None
champion_run = None
champion_rmse = None
champion_data_context = None
expected_champion_version = None
try:
    champion_version = client.get_model_version_by_alias(full_model_name, "Champion")
except MlflowException as exc:
    if "RESOURCE_DOES_NOT_EXIST" not in str(exc):
        raise
    print(f"No current Champion alias found: {exc}")
    promote = True
else:
    expected_champion_version = str(champion_version.version)
    champion_run = mlflow.get_run(champion_version.run_id)
    champion_rmse = champion_run.data.metrics.get("rmse")
    champion_uri = f"models:/{full_model_name}/{champion_version.version}"
    champion_feature_cols = load_feature_cols(
        champion_version.run_id,
        champion_run.data.params,
        "champion",
    )
    eval_feature_cols = sorted(set(challenger_feature_cols) | set(champion_feature_cols))
    features_reader = spark.read
    if features_table_version is not None:
        features_reader = features_reader.option("versionAsOf", int(features_table_version))
    source = features_reader.table(features_table)
    missing_eval_cols = [col for col in eval_feature_cols if col not in source.columns]
    if missing_eval_cols:
        raise ValueError(f"Missing evaluation feature columns in {features_table}: {missing_eval_cols}")
    duplicate_open_time = source.groupBy("open_time").count().filter(F.col("count") > 1).limit(1).collect()
    if duplicate_open_time:
        raise ValueError(
            "Evaluation requires unique open_time rows; "
            f"found duplicate open_time={duplicate_open_time[0]['open_time']}"
        )

    challenger_rows = source.select("open_time", target_col, *challenger_feature_cols).dropna(
        subset=[target_col, *challenger_feature_cols]
    )
    champion_rows = source.select("open_time", target_col, *champion_feature_cols).dropna(
        subset=[target_col, *champion_feature_cols]
    )
    eval_keys = (
        challenger_rows.select("open_time")
        .intersect(champion_rows.select("open_time"))
        .orderBy(F.col("open_time").desc())
        .limit(max_evaluation_rows)
        .orderBy("open_time")
    )
    challenger_eval_pdf = eval_keys.join(challenger_rows, "open_time", "inner").orderBy("open_time").toPandas()
    champion_eval_pdf = eval_keys.join(champion_rows, "open_time", "inner").orderBy("open_time").toPandas()
    if len(challenger_eval_pdf) != len(champion_eval_pdf):
        raise ValueError(
            "Champion/challenger evaluation row mismatch after intersection: "
            f"challenger={len(challenger_eval_pdf)}; champion={len(champion_eval_pdf)}"
        )
    if len(challenger_eval_pdf) < min_evaluation_rows:
        raise ValueError(
            f"Not enough common evaluation rows after dropna: "
            f"{len(challenger_eval_pdf)} < {min_evaluation_rows}"
        )
    challenger_metrics = evaluate_on_rows(challenger_uri, challenger_feature_cols, challenger_eval_pdf)
    champion_metrics = evaluate_on_rows(champion_uri, champion_feature_cols, champion_eval_pdf)
    challenger_fair_rmse = challenger_metrics["rmse"]
    challenger_fair_mae = challenger_metrics["mae"]
    challenger_directional_accuracy = challenger_metrics["directional_accuracy"]
    champion_fair_rmse = champion_metrics["rmse"]
    champion_fair_mae = champion_metrics["mae"]
    champion_directional_accuracy = champion_metrics["directional_accuracy"]
    challenger_eval_rows = len(challenger_eval_pdf)
    champion_eval_rows = len(champion_eval_pdf)
    champion_data_context = {
        "raw_table_version": champion_run.data.params.get("raw_table_version"),
        "features_table_version": champion_run.data.params.get("features_table_version"),
        "feature_config_version": champion_run.data.params.get("feature_config_version"),
        "feature_config_id": champion_run.data.params.get("feature_config_id"),
    }
    print(f"champion_version={champion_version.version}")
    print(f"champion_rmse={champion_rmse}")
    print(f"champion_fair_rmse={champion_fair_rmse}")
    print(f"champion_fair_mae={champion_fair_mae}")
    print(f"champion_directional_accuracy={champion_directional_accuracy}")
    print(f"champion_eval_rows={champion_eval_rows}")
    print(f"champion_data_context={champion_data_context}")
    promote = (
        challenger_fair_rmse < champion_fair_rmse
        and challenger_fair_mae < champion_fair_mae
        and challenger_directional_accuracy >= champion_directional_accuracy
    )

print(f"challenger_run_id={challenger_run_id}")
print(f"challenger_rmse={challenger_rmse}")
print(f"challenger_fair_rmse={challenger_fair_rmse}")
print(f"challenger_fair_mae={challenger_fair_mae}")
print(f"challenger_directional_accuracy={challenger_directional_accuracy}")
print(f"challenger_eval_rows={challenger_eval_rows}")
print(f"training_mode={training_mode}")
print(f"challenger_data_context={challenger_data_context}")

client.log_metric(challenger_run_id, "challenger_fair_rmse", float(challenger_fair_rmse))
if challenger_fair_mae is not None:
    client.log_metric(challenger_run_id, "challenger_fair_mae", float(challenger_fair_mae))
if challenger_directional_accuracy is not None:
    client.log_metric(
        challenger_run_id,
        "challenger_directional_accuracy",
        float(challenger_directional_accuracy),
    )
if champion_fair_rmse is not None:
    client.log_metric(challenger_run_id, "champion_fair_rmse", float(champion_fair_rmse))
if champion_fair_mae is not None:
    client.log_metric(challenger_run_id, "champion_fair_mae", float(champion_fair_mae))
if champion_directional_accuracy is not None:
    client.log_metric(
        challenger_run_id,
        "champion_directional_accuracy",
        float(champion_directional_accuracy),
    )
client.set_tag(challenger_run_id, "promotion_candidate", str(bool(promote)).lower())
client.set_tag(
    challenger_run_id,
    "promotion_expected_champion_version",
    expected_champion_version or "NONE",
)

registered = mlflow.register_model(challenger_uri, full_model_name)
client.set_registered_model_alias(full_model_name, "Challenger", registered.version)
client.set_model_version_tag(full_model_name, registered.version, "promotion_candidate", str(bool(promote)).lower())
client.set_model_version_tag(
    full_model_name,
    registered.version,
    "promotion_expected_champion_version",
    expected_champion_version or "NONE",
)

if promote:
    try:
        current_champion = client.get_model_version_by_alias(full_model_name, "Champion")
        current_champion_version = str(current_champion.version)
    except MlflowException as exc:
        if "RESOURCE_DOES_NOT_EXIST" not in str(exc):
            raise
        current_champion_version = None
    if current_champion_version != expected_champion_version:
        promote = False
        client.set_tag(challenger_run_id, "promotion_lock_conflict", "true")
        client.set_tag(challenger_run_id, "promotion_candidate", "false")
        client.set_model_version_tag(full_model_name, registered.version, "promotion_lock_conflict", "true")
        client.set_model_version_tag(full_model_name, registered.version, "promotion_candidate", "false")
        print(
            "Champion retained because promotion optimistic lock failed: "
            f"expected={expected_champion_version}; actual={current_champion_version}"
        )
    else:
        client.set_registered_model_alias(full_model_name, "Champion", registered.version)
        print(f"PROMOTED Challenger version {registered.version} to Champion")
else:
    print("Champion retained")

# COMMAND ----------

history_schema = T.StructType(
    [
        T.StructField("evaluated_at", T.TimestampType(), True),
        T.StructField("registered_version", T.StringType(), True),
        T.StructField("challenger_run_id", T.StringType(), False),
        T.StructField("champion_run_id", T.StringType(), True),
        T.StructField("expected_champion_version", T.StringType(), True),
        T.StructField("challenger_rmse", T.DoubleType(), True),
        T.StructField("challenger_fair_rmse", T.DoubleType(), True),
        T.StructField("challenger_fair_mae", T.DoubleType(), True),
        T.StructField("challenger_directional_accuracy", T.DoubleType(), True),
        T.StructField("champion_fair_rmse", T.DoubleType(), True),
        T.StructField("champion_fair_mae", T.DoubleType(), True),
        T.StructField("champion_directional_accuracy", T.DoubleType(), True),
        T.StructField("challenger_eval_rows", T.IntegerType(), False),
        T.StructField("champion_eval_rows", T.IntegerType(), False),
        T.StructField("training_mode", T.StringType(), True),
        T.StructField("features_table", T.StringType(), True),
        T.StructField("features_table_version", T.StringType(), True),
        T.StructField("challenger_data_context", T.StringType(), True),
        T.StructField("champion_data_context", T.StringType(), True),
        T.StructField("promoted", T.BooleanType(), False),
    ]
)
history_row = {
    "evaluated_at": None,
    "registered_version": str(registered.version),
    "challenger_run_id": challenger_run_id,
    "champion_run_id": champion_version.run_id if champion_version else None,
    "expected_champion_version": expected_champion_version,
    "challenger_rmse": float(challenger_rmse) if challenger_rmse is not None else None,
    "challenger_fair_rmse": float(challenger_fair_rmse) if challenger_fair_rmse is not None else None,
    "challenger_fair_mae": float(challenger_fair_mae) if challenger_fair_mae is not None else None,
    "challenger_directional_accuracy": (
        float(challenger_directional_accuracy)
        if challenger_directional_accuracy is not None
        else None
    ),
    "champion_fair_rmse": float(champion_fair_rmse) if champion_fair_rmse is not None else None,
    "champion_fair_mae": float(champion_fair_mae) if champion_fair_mae is not None else None,
    "champion_directional_accuracy": (
        float(champion_directional_accuracy)
        if champion_directional_accuracy is not None
        else None
    ),
    "challenger_eval_rows": int(challenger_eval_rows),
    "champion_eval_rows": int(champion_eval_rows),
    "training_mode": training_mode,
    "features_table": features_table,
    "features_table_version": str(features_table_version) if features_table_version is not None else None,
    "challenger_data_context": str(challenger_data_context),
    "champion_data_context": str(champion_data_context) if champion_data_context is not None else None,
    "promoted": bool(promote),
}
history_df = spark.createDataFrame([history_row], history_schema).withColumn(
    "evaluated_at",
    F.current_timestamp(),
)
history_table = f"{catalog}.monitoring.champion_challenger_history"
history_df.write.format("delta").mode("append").saveAsTable(history_table)
print(f"Wrote champion/challenger result to {history_table}")

# COMMAND ----------

display(
    history_df.drop("evaluated_at")
)
