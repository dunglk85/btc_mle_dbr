# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 03 - Optuna Model Training (Regression)
# MAGIC
# MAGIC Notebook huấn luyện mô hình regression với Optuna HPO để dự đoán `target_return_1h` (% thay đổi giá).
# MAGIC
# MAGIC **Models:**
# MAGIC - LightGBM (LGBMRegressor)
# MAGIC - XGBoost (XGBRegressor) — optional
# MAGIC
# MAGIC **Đặc biệt:**
# MAGIC - Time Series Split (không xáo trộn ngẫu nhiên)
# MAGIC - Tự động đọc `selected_features` từ EDA notebook
# MAGIC - MLflow logging đầy đủ

# COMMAND ----------

# MAGIC %pip install optuna lightgbm xgboost

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import mlflow
import optuna
from mlflow.models import infer_signature
from pyspark.sql import functions as F
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# COMMAND ----------

def get_widget(name, default):
    try:
        dbutils.widgets.text(name, str(default))
        return dbutils.widgets.get(name)
    except Exception:
        return str(default)


# --- Widgets ---
catalog = get_widget("catalog", "btc_dev")
model_algo = get_widget("model_algo", "lightgbm")  # "lightgbm" hoặc "xgboost"
n_trials = int(get_widget("n_trials", "30"))
timeout_seconds = int(get_widget("timeout_seconds", "1200"))
n_cv_splits = int(get_widget("n_cv_splits", "5"))
max_decision_age_hours = float(get_widget("max_decision_age_hours", "12"))
expected_trigger_mode = get_widget("expected_trigger_mode", "drift")

# --- Constants ---
features_schema = "features"
features_table = "btc_features"
raw_ref = f"{catalog}.raw.btc_hourly"
features_ref = f"{catalog}.{features_schema}.{features_table}"
config_ref = f"{catalog}.{features_schema}.feature_selection_config"
decisions_ref = f"{catalog}.monitoring.model_refresh_decisions"
training_manifest_ref = f"{catalog}.monitoring.training_dataset_manifests"
target_col = "target_return_1h"

# Validate inputs
assert model_algo in ("lightgbm", "xgboost"), f"Invalid model_algo: {model_algo}"
assert n_trials >= 1, f"n_trials must be >= 1, got {n_trials}"
assert timeout_seconds > 0, f"timeout_seconds must be > 0, got {timeout_seconds}"

experiment_name = f"/Shared/btc_regression_{model_algo}_training"

print("=" * 60)
print(f"RUNNING OPTUNA REGRESSION TRAINING with {model_algo.upper()}")
print(f"features_ref={features_ref}")
print(f"raw_ref={raw_ref}")
print(f"training_manifest_ref={training_manifest_ref}")
print(f"target_col={target_col}")
print(f"n_trials={n_trials}")
print(f"timeout_seconds={timeout_seconds}")
print(f"n_cv_splits={n_cv_splits}")
print(f"max_decision_age_hours={max_decision_age_hours}")
print(f"expected_trigger_mode={expected_trigger_mode}")
print(f"experiment_name={experiment_name}")
print("=" * 60)

# COMMAND ----------

def latest_delta_history(table_ref):
    history = spark.sql(f"DESCRIBE HISTORY {table_ref} LIMIT 1").collect()
    if not history:
        raise ValueError(f"No Delta history found for {table_ref}")
    row = history[0]
    return {
        "table": table_ref,
        "version": int(row["version"]),
        "timestamp": row["timestamp"],
    }


raw_version = latest_delta_history(raw_ref)
features_version = latest_delta_history(features_ref)
try:
    config_version = latest_delta_history(config_ref)
except Exception as exc:
    print(f"WARNING: Could not resolve config table version: {exc}")
    config_version = {"table": config_ref, "version": -1, "timestamp": None}

print(f"raw_table_version={raw_version}")
print(f"features_table_version={features_version}")
print(f"feature_config_version={config_version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Check Training Gate (Skip if not needed)

# COMMAND ----------

skip_reason = None
try:
    latest_decision = (
        spark.table(decisions_ref)
        .orderBy("decision_time", ascending=False)
        .limit(1)
        .collect()
    )
    if not latest_decision:
        skip_reason = "missing training gate decision"
    else:
        decision = latest_decision[0]
        decision_time = decision["decision_time"]
        decision_time_utc = decision_time.replace(tzinfo=timezone.utc)
        decision_age_hours = (
            datetime.now(timezone.utc) - decision_time_utc
        ).total_seconds() / 3600
        print(f"decision_time={decision_time}")
        print(f"decision_age_hours={decision_age_hours:.2f}")
        print(f"decision_trigger_mode={decision['trigger_mode']}")
        print(f"decision_should_retrain={decision['should_retrain']}")
        print(f"decision_reason={decision['reason']}")
        if decision_age_hours > max_decision_age_hours:
            skip_reason = (
                f"stale training gate decision: {decision_age_hours:.2f}h > "
                f"{max_decision_age_hours:.2f}h"
            )
        elif decision["trigger_mode"] != expected_trigger_mode:
            skip_reason = (
                f"unexpected trigger_mode={decision['trigger_mode']}; "
                f"expected={expected_trigger_mode}"
            )
        elif not decision["should_retrain"]:
            skip_reason = decision["reason"]
except Exception as exc:
    skip_reason = f"could not read training gate decision: {exc}"

if skip_reason:
    print(f"SKIP_RETRAIN: {skip_reason}")
    dbutils.jobs.taskValues.set(key="training_status", value="skipped")
    dbutils.notebook.exit("SKIP_RETRAIN")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load Data & Features

# COMMAND ----------

# Đọc selected features từ EDA config
try:
    config_row = spark.table(config_ref).collect()
    if config_row:
        feature_cols = json.loads(config_row[0]["config_value"])
        print(f"Loaded {len(feature_cols)} selected features from EDA config")
    else:
        raise ValueError("Empty config table")
except Exception as e:
    print(f"WARNING: Could not load EDA config ({e}), using default features")
    feature_cols = [
        "return_1h", "return_6h", "return_24h",
        "close_ma7_ratio", "close_ma24_ratio", "close_ma168_ratio",
        "macd", "macd_signal", "macd_hist",
        "rsi_14",
        "atr_14", "atr_ratio", "bb_width",
        "volume_ratio", "log_volume",
        "hl_spread", "oc_change",
        "close_lag_1h", "close_lag_2h", "close_lag_4h", "close_lag_12h", "close_lag_24h",
        "hour", "day_of_week",
        "hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
    ]

print(f"Target column: {target_col}")
print(f"Feature columns ({len(feature_cols)}): {feature_cols}")

# COMMAND ----------

# Load & prepare data
source = spark.table(features_ref).orderBy("open_time")

# Chỉ lấy features thực sự có trong bảng
available_features = [c for c in feature_cols if c in source.columns]
missing_features = [c for c in feature_cols if c not in source.columns]
if missing_features:
    print(f"WARNING: Missing features (skipped): {missing_features}")
feature_cols = available_features

model_df = source.select("open_time", target_col, *feature_cols).dropna()
row_count = model_df.count()
print(f"training_rows_after_dropna={row_count}")
if row_count < 100:
    raise ValueError(f"Not enough training rows in {features_ref}: {row_count}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Time-based Train/Test Split

# COMMAND ----------

pdf = model_df.toPandas().sort_values("open_time").reset_index(drop=True)

# 80/20 split theo thời gian — KHÔNG xáo trộn
split_idx = int(len(pdf) * 0.8)
train = pdf.iloc[:split_idx]
test = pdf.iloc[split_idx:]

X_train = train[feature_cols].values
y_train = train[target_col].values
X_test = test[feature_cols].values
y_test = test[target_col].values

print(f"Train: {len(train)} rows ({train['open_time'].min()} → {train['open_time'].max()})")
print(f"Test:  {len(test)} rows ({test['open_time'].min()} → {test['open_time'].max()})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Define Optuna Objective

# COMMAND ----------

# Time Series Cross-Validation — đảm bảo không data leakage
tscv = TimeSeriesSplit(n_splits=n_cv_splits)

mlflow.set_experiment(experiment_name)


def create_model(trial, algo):
    """Tạo model với hyperparameters từ Optuna trial."""
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

    elif algo == "xgboost":
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


def objective(trial):
    """Optuna objective function với Time Series Cross-Validation."""
    model, params = create_model(trial, model_algo)

    cv_scores = []
    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

        model_fold, _ = create_model(trial, model_algo)
        model_fold.fit(X_fold_train, y_fold_train)
        preds = model_fold.predict(X_fold_val)
        score = mean_squared_error(y_fold_val, preds) ** 0.5  # RMSE

        cv_scores.append(score)

    mean_score = np.mean(cv_scores)

    # Log trial to MLflow
    with mlflow.start_run(
        run_name=f"optuna_{model_algo}_regression_trial_{trial.number}",
        nested=True,
    ):
        mlflow.log_param("model_algo", model_algo)
        mlflow.log_param("task_type", "regression")
        mlflow.log_param("trial_number", trial.number)
        mlflow.log_params(params)
        mlflow.log_metric("cv_rmse", mean_score)

    return mean_score

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Run Optuna Optimization

# COMMAND ----------

with mlflow.start_run(
    run_name=f"optuna_{model_algo}_regression_parent"
) as parent_run:

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout_seconds)

    # --- Train final model với best params trên toàn bộ train set ---
    best_trial = study.best_trial
    best_model, best_params = create_model(best_trial, model_algo)
    best_model.fit(X_train, y_train)

    # --- Evaluate trên test set ---
    best_preds = best_model.predict(X_test)
    rmse = mean_squared_error(y_test, best_preds) ** 0.5
    mae = mean_absolute_error(y_test, best_preds)
    r2 = r2_score(y_test, best_preds)
    mape = float((np.abs(y_test - best_preds) / np.abs(y_test).clip(min=1e-10)).mean())

    # Directional accuracy is derived from regression predictions, not a classifier.
    pred_direction = (best_preds > 0).astype(int)
    actual_direction = (y_test > 0).astype(int)
    directional_accuracy = (pred_direction == actual_direction).mean()

    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mape", mape)
    mlflow.log_metric("directional_accuracy", directional_accuracy)

    print(f"\n{'='*60}")
    print(f"REGRESSION RESULTS ({model_algo.upper()})")
    print(f"{'='*60}")
    print(f"RMSE:                  {rmse:.6f}")
    print(f"MAE:                   {mae:.6f}")
    print(f"R²:                    {r2:.4f}")
    print(f"MAPE:                  {mape:.6f}")
    print(f"Directional Accuracy:  {directional_accuracy:.4f}")

    # --- Log common params & model ---
    mlflow.log_param("model_algo", model_algo)
    mlflow.log_param("task_type", "regression")
    mlflow.log_param("training_mode", "optuna")
    mlflow.log_param("raw_table", raw_ref)
    mlflow.log_param("raw_table_version", raw_version["version"])
    mlflow.log_param("features_table", features_ref)
    mlflow.log_param("features_table_version", features_version["version"])
    mlflow.log_param("feature_config_table", config_ref)
    if config_version["version"] >= 0:
        mlflow.log_param("feature_config_version", config_version["version"])
    mlflow.log_param("n_trials_requested", n_trials)
    mlflow.log_param("n_trials_completed", len(study.trials))
    mlflow.log_param("n_cv_splits", n_cv_splits)
    mlflow.log_param("n_features", len(feature_cols))
    mlflow.log_param("train_rows", len(train))
    mlflow.log_param("test_rows", len(test))
    mlflow.log_params(best_params)

    # Log feature list
    mlflow.log_text(json.dumps(feature_cols, indent=2), "feature_cols.json")
    mlflow.log_text(
        json.dumps(
            {
                "raw": {"table": raw_ref, "version": raw_version["version"], "timestamp": str(raw_version["timestamp"])},
                "features": {"table": features_ref, "version": features_version["version"], "timestamp": str(features_version["timestamp"])},
                "feature_config": {"table": config_ref, "version": config_version["version"], "timestamp": str(config_version["timestamp"])},
                "target_col": target_col,
                "feature_cols": feature_cols,
                "train_start_time": str(train["open_time"].min()),
                "train_end_time": str(train["open_time"].max()),
                "test_start_time": str(test["open_time"].min()),
                "test_end_time": str(test["open_time"].max()),
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
            },
            indent=2,
        ),
        "training_dataset_manifest.json",
    )

    # Log model
    signature = infer_signature(
        pd.DataFrame(X_train, columns=feature_cols),
        best_model.predict(X_train),
    )
    if model_algo == "lightgbm":
        mlflow.lightgbm.log_model(best_model, "model", signature=signature)
    else:
        mlflow.xgboost.log_model(best_model, "model", signature=signature)

    run_id = parent_run.info.run_id

dbutils.jobs.taskValues.set(key="training_status", value="trained")
dbutils.jobs.taskValues.set(key="run_id", value=run_id)
dbutils.jobs.taskValues.set(key="task_type", value="regression")

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
        train_start_time TIMESTAMP,
        train_end_time TIMESTAMP,
        test_start_time TIMESTAMP,
        test_end_time TIMESTAMP,
        train_rows BIGINT,
        test_rows BIGINT,
        n_features BIGINT,
        feature_cols_json STRING
    )
    USING DELTA
""")

manifest_df = spark.createDataFrame(
    [
        {
            "run_id": run_id,
            "model_algo": model_algo,
            "target_col": target_col,
            "raw_table": raw_ref,
            "raw_table_version": raw_version["version"],
            "features_table": features_ref,
            "features_table_version": features_version["version"],
            "feature_config_table": config_ref,
            "feature_config_version": config_version["version"],
            "train_start_time": train["open_time"].min().to_pydatetime(),
            "train_end_time": train["open_time"].max().to_pydatetime(),
            "test_start_time": test["open_time"].min().to_pydatetime(),
            "test_end_time": test["open_time"].max().to_pydatetime(),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "n_features": len(feature_cols),
            "feature_cols_json": json.dumps(feature_cols),
        }
    ]
).withColumn("created_at", F.current_timestamp())

manifest_df.select(
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
    "train_start_time",
    "train_end_time",
    "test_start_time",
    "test_end_time",
    "train_rows",
    "test_rows",
    "n_features",
    "feature_cols_json",
).write.mode("append").saveAsTable(training_manifest_ref)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Results Summary

# COMMAND ----------

print(f"run_id={run_id}")
print("task_type=regression")
print(f"model_algo={model_algo}")
print(f"best_params={best_params}")
print(f"n_trials_completed={len(study.trials)}")
print(f"best_trial_value={study.best_value:.6f}")

# COMMAND ----------

# Feature Importance Plot
if hasattr(best_model, "feature_importances_"):
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": best_model.feature_importances_,
    }).sort_values("importance", ascending=True)

    display(spark.createDataFrame(
        importance_df.sort_values("importance", ascending=False)
    ))

# COMMAND ----------

# Summary table
summary_data = {
    "run_id": run_id,
    "task_type": "regression",
    "model_algo": model_algo,
    "rmse": float(rmse),
    "mae": float(mae),
    "r2": float(r2),
    "mape": float(mape),
    "directional_accuracy": float(directional_accuracy),
    "n_trials_requested": int(n_trials),
    "n_trials_completed": int(len(study.trials)),
    "train_rows": int(len(train)),
    "test_rows": int(len(test)),
    "n_features": len(feature_cols),
}

display(spark.createDataFrame([summary_data]))
