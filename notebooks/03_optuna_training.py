# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 03 - Optuna Model Training And Best Challenger Selection
# MAGIC Train one regression candidate on the latest feature table and expose its MLflow run through task values. Databricks Jobs runs model candidates in parallel tasks.

# COMMAND ----------

# MAGIC %pip install optuna lightgbm xgboost scikit-learn shap

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import json
from datetime import datetime

import mlflow
import numpy as np
import optuna
import pandas as pd
import shap
from mlflow.models import infer_signature
from pyspark.sql import functions as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

# COMMAND ----------

def get_widget(name, default):
    try:
        dbutils.widgets.text(name, str(default))
        return dbutils.widgets.get(name)
    except Exception:
        return str(default)


catalog = get_widget("catalog", "btc_simply")
model_algo = get_widget("model_algo", "lightgbm")
n_trials = int(get_widget("n_trials", "30"))
timeout_seconds = int(get_widget("timeout_seconds", "1200"))
n_cv_splits = int(get_widget("n_cv_splits", "5"))
train_fraction = float(get_widget("train_fraction", "0.8"))
allow_default_feature_fallback = get_widget("allow_default_feature_fallback", "false").lower() == "true"
allow_missing_feature_skip = get_widget("allow_missing_feature_skip", "false").lower() == "true"
enable_shap_explanation = get_widget("enable_shap_explanation", "true").lower() == "true"
shap_sample_rows = int(get_widget("shap_sample_rows", "500"))

assert model_algo in {"lightgbm", "xgboost", "random_forest"}, f"Invalid model_algo: {model_algo}"
assert n_trials >= 1, f"n_trials must be >= 1, got {n_trials}"
assert timeout_seconds > 0, f"timeout_seconds must be > 0, got {timeout_seconds}"
assert 0.0 < train_fraction < 1.0, f"train_fraction must be between 0 and 1, got {train_fraction}"
assert shap_sample_rows >= 1, f"shap_sample_rows must be >= 1, got {shap_sample_rows}"

features_schema = "features"
raw_ref = f"{catalog}.raw.btc_hourly"
features_ref = f"{catalog}.{features_schema}.btc_features"
config_ref = f"{catalog}.{features_schema}.feature_selection_config"
training_manifest_ref = f"{catalog}.monitoring.training_dataset_manifests"
model_explanations_ref = f"{catalog}.monitoring.model_explanations"
target_col = "target_return_1h"
experiment_name = "/Shared/btc_regression_training"

print("RUNNING OPTUNA REGRESSION TRAINING AND BEST CHALLENGER SELECTION")
print(f"model_algo={model_algo}")
print(f"features_ref={features_ref}")
print(f"raw_ref={raw_ref}")
print(f"config_ref={config_ref}")
print(f"training_manifest_ref={training_manifest_ref}")
print(f"target_col={target_col}")
print(f"n_trials={n_trials}")
print(f"timeout_seconds={timeout_seconds}")

# COMMAND ----------

def latest_delta_history(table_ref):
    history = spark.sql(f"DESCRIBE HISTORY {table_ref} LIMIT 1").collect()
    if not history:
        raise ValueError(f"No Delta history found for {table_ref}")
    row = history[0]
    return {"table": table_ref, "version": int(row["version"]), "timestamp": row["timestamp"]}


raw_version = latest_delta_history(raw_ref)
features_version = latest_delta_history(features_ref)
try:
    config_version = latest_delta_history(config_ref)
except Exception as exc:
    print(f"WARNING: Could not resolve config table version: {exc}")
    config_version = {"table": config_ref, "version": -1, "timestamp": None}

# COMMAND ----------

try:
    config_row = (
        spark.table(config_ref)
        .filter(F.col("config_key") == "selected_features")
        .filter(F.col("is_active") == True)
        .orderBy(F.col("created_at").desc())
        .limit(1)
        .collect()
    )
    if not config_row:
        raise ValueError("No active selected_features config found")
    feature_cols = json.loads(config_row[0]["config_value"])
    config_dict = config_row[0].asDict()
    feature_config_id = config_dict.get("config_id") or config_dict.get("config_version") or -1
except Exception as exc:
    if not allow_default_feature_fallback:
        raise ValueError(
            f"Could not load active feature config from {config_ref}: {exc}. "
            "Run 02_feature_engineering or set allow_default_feature_fallback=true explicitly."
        ) from exc
    print(f"WARNING: Using default feature fallback because active config could not be loaded: {exc}")
    feature_config_id = -1
    feature_cols = [
        "return_1h", "return_6h", "return_24h",
        "close_ma7_ratio", "close_ma24_ratio", "close_ma168_ratio",
        "macd", "macd_signal", "macd_hist",
        "rsi_14", "atr_14", "atr_ratio", "bb_width",
        "volume_ratio", "log_volume", "hl_spread", "oc_change",
        "close_lag_1h", "close_lag_2h", "close_lag_4h", "close_lag_12h", "close_lag_24h",
        "hour", "day_of_week", "hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
    ]

source = spark.table(features_ref).orderBy("open_time")
available_features = [col for col in feature_cols if col in source.columns]
missing_features = [col for col in feature_cols if col not in source.columns]
if missing_features:
    if not allow_missing_feature_skip:
        raise ValueError(f"Active feature config references missing columns in {features_ref}: {missing_features}")
    print(f"WARNING: Missing features skipped: {missing_features}")
feature_cols = available_features

model_df = source.select("open_time", target_col, *feature_cols).dropna()
row_count = model_df.count()
print(f"training_rows_after_dropna={row_count}")
if row_count < 100:
    raise ValueError(f"Not enough training rows in {features_ref}: {row_count}")

duplicate_open_time_count = model_df.groupBy("open_time").count().filter(F.col("count") > 1).count()
if duplicate_open_time_count > 0:
    raise ValueError(f"Training data contains {duplicate_open_time_count} duplicate open_time values")

pdf = model_df.toPandas().sort_values("open_time").reset_index(drop=True)
split_idx = int(len(pdf) * train_fraction)
train = pdf.iloc[:split_idx]
test = pdf.iloc[split_idx:]
X_train = train[feature_cols].values
y_train = train[target_col].values
X_test = test[feature_cols].values
y_test = test[target_col].values
tscv = TimeSeriesSplit(n_splits=n_cv_splits)

print(f"feature_cols={feature_cols}")
print(f"train_rows={len(train)}")
print(f"test_rows={len(test)}")

# COMMAND ----------

def create_model(trial, algo):
    if algo == "lightgbm":
        from lightgbm import LGBMRegressor

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state": 42,
            "verbose": -1,
        }
        return LGBMRegressor(**params), params

    if algo == "random_forest":
        from sklearn.ensemble import RandomForestRegressor

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 24),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.8, 1.0]),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "random_state": 42,
            "n_jobs": -1,
        }
        return RandomForestRegressor(**params), params

    from xgboost import XGBRegressor

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "random_state": 42,
        "verbosity": 0,
    }
    return XGBRegressor(**params), params


def train_candidate(algo):
    mlflow.set_experiment(experiment_name)

    def objective(trial):
        scores = []
        for train_idx, val_idx in tscv.split(X_train):
            model, _ = create_model(trial, algo)
            model.fit(X_train[train_idx], y_train[train_idx])
            preds = model.predict(X_train[val_idx])
            scores.append(mean_squared_error(y_train[val_idx], preds) ** 0.5)
        return float(np.mean(scores))

    with mlflow.start_run(run_name=f"optuna_{algo}_regression_parent") as parent_run:
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        )
        study.optimize(objective, n_trials=n_trials, timeout=timeout_seconds)

        best_model, best_params = create_model(study.best_trial, algo)
        best_model.fit(X_train, y_train)
        preds = best_model.predict(X_test)
        rmse = mean_squared_error(y_test, preds) ** 0.5
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        mape = float((np.abs(y_test - preds) / np.abs(y_test).clip(min=1e-10)).mean())
        directional_accuracy = float(((preds > 0).astype(int) == (y_test > 0).astype(int)).mean())

        mlflow.log_param("model_algo", algo)
        mlflow.log_param("task_type", "regression")
        mlflow.log_param("target_col", target_col)
        mlflow.log_param("training_mode", "optuna")
        mlflow.log_param("raw_table", raw_ref)
        mlflow.log_param("raw_table_version", raw_version["version"])
        mlflow.log_param("features_table", features_ref)
        mlflow.log_param("features_table_version", features_version["version"])
        mlflow.log_param("feature_config_table", config_ref)
        mlflow.log_param("feature_config_version", config_version["version"])
        mlflow.log_param("feature_config_id", feature_config_id)
        mlflow.log_param("n_trials_requested", n_trials)
        mlflow.log_param("n_trials_completed", len(study.trials))
        mlflow.log_param("n_cv_splits", n_cv_splits)
        mlflow.log_param("train_fraction", train_fraction)
        mlflow.log_param("split_idx", split_idx)
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("train_rows", len(train))
        mlflow.log_param("test_rows", len(test))
        mlflow.log_params(best_params)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mape", mape)
        mlflow.log_metric("directional_accuracy", directional_accuracy)

        mlflow.log_text(json.dumps(feature_cols, indent=2), "feature_cols.json")
        mlflow.log_text(
            json.dumps(
                {
                    "raw": {"table": raw_ref, "version": raw_version["version"], "timestamp": str(raw_version["timestamp"])},
                    "features": {"table": features_ref, "version": features_version["version"], "timestamp": str(features_version["timestamp"])},
                    "feature_config": {"table": config_ref, "version": config_version["version"], "config_id": feature_config_id, "timestamp": str(config_version["timestamp"])},
                    "target_col": target_col,
                    "feature_cols": feature_cols,
                    "train_start_time": str(train["open_time"].min()),
                    "train_end_time": str(train["open_time"].max()),
                    "test_start_time": str(test["open_time"].min()),
                    "test_end_time": str(test["open_time"].max()),
                    "train_rows": int(len(train)),
                    "test_rows": int(len(test)),
                    "train_fraction": float(train_fraction),
                    "split_idx": int(split_idx),
                },
                indent=2,
            ),
            "training_dataset_manifest.json",
        )

        explanation_rows = []
        if hasattr(best_model, "feature_importances_"):
            importance_df = pd.DataFrame({"feature": feature_cols, "importance": best_model.feature_importances_}).sort_values("importance", ascending=False)
            mlflow.log_text(importance_df.to_json(orient="records", indent=2), "model_explanation/feature_importance.json")
            explanation_rows.extend(
                {
                    "run_id": parent_run.info.run_id,
                    "model_algo": algo,
                    "explanation_type": "feature_importance",
                    "feature": row["feature"],
                    "importance": float(row["importance"]),
                    "mean_abs_shap": None,
                    "mean_shap": None,
                    "sample_rows": None,
                    "features_table_version": int(features_version["version"]),
                    "feature_config_id": int(feature_config_id),
                }
                for row in importance_df.to_dict(orient="records")
            )

        if enable_shap_explanation:
            shap_sample = test[feature_cols].head(shap_sample_rows).astype("float64")
            if len(shap_sample) > 0:
                explainer = shap.TreeExplainer(best_model)
                shap_values = explainer.shap_values(shap_sample)
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]
                shap_values = np.asarray(shap_values, dtype="float64")
                shap_summary_df = pd.DataFrame(
                    {
                        "feature": feature_cols,
                        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
                        "mean_shap": shap_values.mean(axis=0),
                    }
                ).sort_values("mean_abs_shap", ascending=False)
                mlflow.log_text(shap_summary_df.to_json(orient="records", indent=2), "model_explanation/shap_summary.json")
                explanation_rows.extend(
                    {
                        "run_id": parent_run.info.run_id,
                        "model_algo": algo,
                        "explanation_type": "shap_summary",
                        "feature": row["feature"],
                        "importance": None,
                        "mean_abs_shap": float(row["mean_abs_shap"]),
                        "mean_shap": float(row["mean_shap"]),
                        "sample_rows": int(len(shap_sample)),
                        "features_table_version": int(features_version["version"]),
                        "feature_config_id": int(feature_config_id),
                    }
                    for row in shap_summary_df.to_dict(orient="records")
                )

        if explanation_rows:
            spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.monitoring")
            spark.createDataFrame(explanation_rows).withColumn("created_at", F.current_timestamp()).write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(model_explanations_ref)

        signature = infer_signature(pd.DataFrame(X_train, columns=feature_cols), best_model.predict(X_train))
        if algo == "lightgbm":
            mlflow.lightgbm.log_model(best_model, "model", signature=signature)
        elif algo == "random_forest":
            mlflow.sklearn.log_model(best_model, "model", signature=signature)
        else:
            mlflow.xgboost.log_model(best_model, "model", signature=signature)

        return {
            "run_id": parent_run.info.run_id,
            "model_algo": algo,
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
            "mape": float(mape),
            "directional_accuracy": directional_accuracy,
            "n_trials_completed": int(len(study.trials)),
            "best_params": best_params,
        }


result = train_candidate(model_algo)

print(f"trained_model_algo={result['model_algo']}")
print(f"run_id={result['run_id']}")
print(f"rmse={result['rmse']}")
print(f"mae={result['mae']}")
print(f"directional_accuracy={result['directional_accuracy']}")

dbutils.jobs.taskValues.set(key="training_status", value="trained")
dbutils.jobs.taskValues.set(key="run_id", value=result["run_id"])
dbutils.jobs.taskValues.set(key="task_type", value="regression")
dbutils.jobs.taskValues.set(key="model_algo", value=result["model_algo"])

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.monitoring")
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {training_manifest_ref} (
        created_at TIMESTAMP,
        run_id STRING,
        model_algo STRING,
        target_col STRING,
        raw_table STRING,
        raw_table_version BIGINT,
        features_table STRING,
        features_table_version BIGINT,
        feature_config_table STRING,
        feature_config_version BIGINT,
        feature_config_id BIGINT,
        train_start_time TIMESTAMP,
        train_end_time TIMESTAMP,
        test_start_time TIMESTAMP,
        test_end_time TIMESTAMP,
        train_rows BIGINT,
        test_rows BIGINT,
        train_fraction DOUBLE,
        split_idx BIGINT,
        n_features BIGINT,
        feature_cols_json STRING
    )
    USING DELTA
""")

for column_def in ["feature_config_id BIGINT", "train_fraction DOUBLE", "split_idx BIGINT"]:
    try:
        spark.sql(f"ALTER TABLE {training_manifest_ref} ADD COLUMNS ({column_def})")
    except Exception as exc:
        print(f"manifest_column_add_skipped={column_def}: {exc}")

manifest_rows = [{
    "run_id": result["run_id"],
    "model_algo": result["model_algo"],
    "target_col": target_col,
    "raw_table": raw_ref,
    "raw_table_version": raw_version["version"],
    "features_table": features_ref,
    "features_table_version": features_version["version"],
    "feature_config_table": config_ref,
    "feature_config_version": config_version["version"],
    "feature_config_id": feature_config_id,
    "train_start_time": train["open_time"].min().to_pydatetime(),
    "train_end_time": train["open_time"].max().to_pydatetime(),
    "test_start_time": test["open_time"].min().to_pydatetime(),
    "test_end_time": test["open_time"].max().to_pydatetime(),
    "train_rows": int(len(train)),
    "test_rows": int(len(test)),
    "train_fraction": float(train_fraction),
    "split_idx": int(split_idx),
    "n_features": len(feature_cols),
    "feature_cols_json": json.dumps(feature_cols),
}]

spark.createDataFrame(manifest_rows).withColumn("created_at", F.current_timestamp()).select(
    "created_at",
    "run_id",
    "model_algo",
    "target_col",
    "raw_table",
    "raw_table_version",
    "features_table",
    "features_table_version",
    "feature_config_table",
    "feature_config_version",
    "feature_config_id",
    "train_start_time",
    "train_end_time",
    "test_start_time",
    "test_end_time",
    "train_rows",
    "test_rows",
    "train_fraction",
    "split_idx",
    "n_features",
    "feature_cols_json",
).write.mode("append").option("mergeSchema", "true").saveAsTable(training_manifest_ref)

# COMMAND ----------

display(spark.createDataFrame([result]))
